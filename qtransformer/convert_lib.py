import tensorflow as tf
import numpy as np
import os
import itertools
from tensor2tensor.utils import registry


def make_weight_map(scopes, weight_name, mask_name, sharded=False, num_shards=1):
    return {scope: (weight_name, mask_name, sharded, num_shards) for scope in scopes}


# Embeddings
embedding_scopes = ['transformer/symbol_modality_33708_512/shared']
embedding_weight_map = make_weight_map(embedding_scopes, 'weights', 'weights_mask',
        sharded=True, num_shards=16)

# FFN weights
ffn_scopes = list(map(
        lambda x: 'transformer/body/{}/layer_{}/{}'.format(*x),
        itertools.product(['encoder', 'decoder'], range(6), ['ffn/conv1', 'ffn/conv2'])))
ffn_weight_map = make_weight_map(ffn_scopes, 'kernel', 'kernel_mask', sharded=False)

# Encoder attention weights
encoder_attn_scopes = list(map(
        lambda x: 'transformer/body/{}/layer_{}/{}/multihead_attention/{}'.format(*x),
        itertools.product(['encoder'], range(6), ['self_attention'], ['q', 'k', 'v', 'output_transform'])))
encoder_attn_weight_map = make_weight_map(encoder_attn_scopes, 'kernel', 'kernel_mask', sharded=False)

# Decoder attention weights
decoder_attn_scopes = list(map(
        lambda x: 'transformer/body/{}/layer_{}/{}/multihead_attention/{}'.format(*x),
        itertools.product(['decoder'], range(6), ['self_attention', 'encdec_attention'], ['q', 'k', 'v', 'output_transform'])))
decoder_attn_weight_map = make_weight_map(decoder_attn_scopes, 'kernel', 'kernel_mask', sharded=False)

# Full weight map
weight_map = dict(list(embedding_weight_map.items())
        + list(ffn_weight_map.items())
        + list(encoder_attn_weight_map.items())
        + list(decoder_attn_weight_map.items()))


def scoped_names(name, scope=None, sharded=False, num_shards=1):
    assert scope is None or scope[-1] != '/'
    if scope is None:
        scope = ''
    else:
        scope = scope + '/'

    if sharded:
        return ['{}{}_{}'.format(scope, name, i) for i in range(num_shards)]
    else:
        return ['{}{}'.format(scope, name)]


def get_tensor_from_ckpt(reader, name, scope=None, sharded=False, num_shards=1):
    names = scoped_names(name, scope, sharded, num_shards)
    tensors = map(reader.get_tensor, names)
    return tensors


def reader_get_tensor_or_none(reader, name):
    try:
        return reader.get_tensor(name)
    except tf.errors.NotFoundError:
        return None


def get_all_tensors_from_ckpt(reader):
    embedding_vars = [
        'transformer/symbol_modality_33708_512/shared/weights_{}'.format(i)
         for i in range(16)
    ]

    ffn_vars = [
        'transformer/body/{name}/layer_{idx}/{op}/kernel'.format(
            name=layer_name, idx=layer_idx, op=layer_op)
        for layer_name, layer_idx, layer_op in itertools.product(
            ['encoder', 'decoder'], range(6), ['ffn/conv1', 'ffn/conv2'])
    ]

    encoder_attn_vars = [
        'transformer/body/{name}/layer_{idx}/{attn}/multihead_attention/{transform}/kernel'.format(
            name=layer_name, idx=layer_idx, attn=attn_type, transform=transform)
        for layer_name, layer_idx, attn_type, transform in itertools.product(
            ['encoder'], range(6), ['self_attention'], ['q', 'k', 'v', 'output_transform'])
    ]

    decoder_attn_vars = [
        'transformer/body/{name}/layer_{idx}/{attn}/multihead_attention/{transform}/kernel'.format(
            name=layer_name, idx=layer_idx, attn=attn_type, transform=transform)
        for layer_name, layer_idx, attn_type, transform in itertools.product(
            ['decoder'], range(6), ['self_attention', 'encdec_attention'],
            ['q', 'k', 'v', 'output_transform'])
    ]

    var_names = embedding_vars + ffn_vars + encoder_attn_vars + decoder_attn_vars
    return {name: reader.get_tensor(name) for name in var_names}


def get_masks_from_ckpt(reader, threshold_percentile=0.4):
    # Get all of the tensors from the checkpoint
    all_tensors = []
    for scope, (weight_name, mask_name, sharded, num_shards) in weight_map.items():
        # Expand the sharded name into a list of full names
        weight_names = scoped_names(weight_name, scope, sharded, num_shards)
        mask_names = scoped_names(mask_name, scope, sharded, num_shards)

        # Get the weights for the shard
        weights = map(reader.get_tensor, weight_names)

        # Try to get the mask for the shard
        masks = map(lambda n: reader_get_tensor_or_none(reader, n), mask_names)

        all_tensors.extend(zip(weight_names, weights, mask_names, masks))
        break

    # Compute the threshold over the unmasked weights
    unmasked_weights = []
    mask_count = 0
    total_count = 0
    for _, weight, _, mask in all_tensors:
        if mask is not None:
            unmasked_weights.append(weight[mask])
            mask_count += np.count_nonzero(mask)
        else:
            unmasked_weights.append(weight)

        total_count += weight.size

    unpruned_percentile = mask_count / total_count
    percentile = 1 - ((1 - threshold_percentile) / unpruned_percentile)
    threshold = np.percentile(np.abs(np.hstack(unmasked_weights)),
            np.max([percentile, 0]) * 100)

    # Update the masks with the locations that have been pruned
    updated_tensors = []
    for weight_name, weight, mask_name, mask in all_tensors:
        mask = np.logical_or(mask if mask is not None else False, np.abs(weight) < threshold)
        updated_tensors.append((weight_name, weight, mask_name, mask))

    return updated_tensors


def get_threshold_from_ckpt(reader, threshold_percentile):
    tensor_map = get_all_tensors_from_ckpt(reader)
    threshold = np.percentile(np.abs(np.hstack([
        tensor.ravel() for tensor in tensor_map.values()
    ])), threshold_percentile * 100)

    return threshold


def make_init_op(name, value, scope=None, dtype=None):
    dtype = value.dtype if dtype is None else dtype
    with tf.variable_scope(scope, reuse=True):
        var = tf.get_variable(name, dtype=dtype)
        return tf.assign(var, value)


def create_initializers(names, values, scope=None, dtype=None):
    return [make_init_op(name, value, scope=scope, dtype=dtype)
            for name, value in zip(names, values)]


def create_initializer(name, value, scope=None, dtype=None, sharded=False, num_shards=1):
    # Want to just make sharded names so no scope
    names = scoped_names(name, sharded=sharded, num_shards=num_shards)
    initializers = create_initializers(names, value, scope=scope, dtype=dtype)
    if sharded and num_shards > 1:
        return tf.group(initializers)
    return initializers[0]


def prune_masks(weights, threshold):
    return [np.abs(w) >= threshold for w in weights]


def make_prune_initializers_from_ckpt(graph, reader, weight_map, threshold_percentile=0.4):
    threshold = get_threshold_from_ckpt(reader, threshold_percentile)

    initializers = []
    with graph.as_default():
        for scope, [weight_name, mask_name, sharded, num_shards] in weight_map.items():
            weight_names = scoped_names(weight_name, scope=scope,
                                        sharded=sharded, num_shards=num_shards)
            weights = map(reader.get_tensor, weight_names)
            weights_mask = prune_masks(weights, threshold)

            initializers.append(create_initializer(
                mask_name, weights_mask, scope=scope, dtype=tf.bool,
                sharded=sharded, num_shards=num_shards))

    return initializers


def make_prune_saver_from_graph(graph):
    mask_vars = set(graph.get_collection('masks'))
    graph_vars = set(graph.get_collection(tf.GraphKeys.VARIABLES))
    train_vars = set(
            graph.get_collection(tf.GraphKeys.VARIABLES, scope='training') +
            graph.get_collection(tf.GraphKeys.VARIABLES, scope='losses_avg'))

    return tf.train.Saver(graph_vars - mask_vars - train_vars)


def prune_checkpoint(hparams, ckpt_dir, threshold_percentile=0.4, old_ckpt=None, new_ckpt=None):
    # Setup the checkpoints to some reasonable defaults
    if old_ckpt is None:
        old_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if new_ckpt is None:
        new_ckpt = os.path.join(ckpt_dir, 'pruned_{}'.format(int(threshold_percentile * 100)))

    # Create a reader for the old checkpoint
    reader = tf.train.NewCheckpointReader(old_ckpt)

    # Create the graph and the initializers
    graph = make_transformer(hparams)
    saver = make_prune_saver_from_graph(graph)
    inits = make_prune_initializers_from_ckpt(graph, reader, weight_map, threshold_percentile)

    # Create a new saver for the initialized graph
    new_saver = tf.train.Saver(graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, old_ckpt)
        sess.run(inits)

        # Save the now initialized graph to disk
        new_saver.save(sess, new_ckpt, global_step=tf.train.get_global_step(graph))
    return graph


def quantize_weights(ws, num_codes, sharded=False):
    if sharded:
        ws_full = np.vstack(ws).flatten()
    else:
        ws_full = ws.flatten()

    # Indexes are the bin that each weight falls in
    bins = np.linspace(0, 100, num_codes + 1)
    pers = np.percentile(ws_full, bins)
    pers[-1] += 1000 # Increase the range of the last bucket so digitize does the right thing
    if sharded:
        idxs = [np.digitize(w, pers) - 1 for w in ws]
    else:
        idxs = np.digitize(ws, pers) - 1

    # Quantized weights are the center of each bin
    codes = np.percentile(ws_full, bins[:-1] + np.diff(bins)/2)

    return idxs, codes


def quantize_scope(reader, scope, weights_name, idxs_name, codebook_name,
        num_codes, sharded=False, num_shards=1):
    if sharded:
        ws = [reader.get_tensor(scope + '/' + weights_name + '_{}'.format(i))
            for i in range(num_shards)]
    else:
        ws = reader.get_tensor(scope + '/' + weights_name)

    idxs, codes = quantize_weights(ws, num_codes=num_codes, sharded=sharded)

    initializers = []
    with tf.variable_scope(scope, reuse=True):
        initializers.append(tf.assign(tf.get_variable(codebook_name), codes).op)

        if sharded:
            for i in range(num_shards):
                initializers.append(tf.assign(
                    tf.get_variable(idxs_name + '_{}'.format(i), dtype=tf.int32), idxs[i]).op)
        else:
            initializers.append(tf.assign(
                tf.get_variable(idxs_name, dtype=tf.int32), idxs).op)

    return initializers


def make_transformer(hparams, mode=tf.estimator.ModeKeys.TRAIN, graph=None):
    if graph is None:
        graph = tf.Graph()

    with graph.as_default():
        # Build the graph for the problem
        # Create the input
        problem = hparams.problem_instances[0]
        features, labels = problem.input_fn(mode, hparams)

        # Create the model
        model_cls = registry.model('transformer')
        estimator_spec = model_cls.estimator_model_fn(hparams, features, labels, mode)

    return graph
