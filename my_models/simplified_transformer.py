# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer model from "Attention Is All You Need".

The Transformer model consists of an encoder and a decoder. Both are stacks
of self-attention layers followed by feed-forward layers. This model yields
good results on a number of problems, especially in NLP and machine translation.

See "Attention Is All You Need" (https://arxiv.org/abs/1706.03762) for the full
description of the model and the results obtained with its early version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import transformer_layers
from tensor2tensor.utils import beam_search
from tensor2tensor.utils import mlperf_log
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf

from tensorflow.python.ops import inplace_ops
from tensorflow.python.util import nest
from tensor2tensor.models.transformer import transformer_base

# Alias some commonly reused layers, here and elsewhere.
transformer_prepare_encoder = transformer_layers.transformer_prepare_encoder
transformer_encoder = transformer_layers.transformer_encoder
transformer_ffn_layer = transformer_layers.transformer_ffn_layer


def simplehead_attention(query_antecedent,
                         memory_antecedent,
                         bias,
                         total_key_depth,
                         total_value_depth,
                         output_depth,
                         num_heads,
                         dropout_rate,
                         attention_type="dot_product",
                         max_relative_position=None,
                         heads_share_relative_embedding=False,
                         add_relative_to_values=False,
                         image_shapes=None,
                         block_length=128,
                         block_width=128,
                         cache=None,
                         gap_size=0,
                         num_memory_blocks=2,
                         name="simplehead_attention",
                         save_weights_to=None,
                         make_image_summary=True,
                         dropout_broadcast_dims=None,
                         vars_3d=False,
                         **kwargs):
    """Multihead scaled-dot-product attention with input/output transformations.
    Args:
      query_antecedent: a Tensor with shape [batch, length_q, channels]
      memory_antecedent: a Tensor with shape [batch, length_m, channels] or None
      bias: bias Tensor (see attention_bias())
      total_key_depth: an integer
      total_value_depth: an integer
      output_depth: an integer
      num_heads: an integer dividing total_key_depth and total_value_depth
      dropout_rate: a floating point number
      attention_type: a string, either "dot_product", "dot_product_relative",
                      "local_mask_right", "local_unmasked", "masked_dilated_1d",
                      "unmasked_dilated_1d", graph, or any attention function
                      with the signature (query, key, value, **kwargs)
      max_relative_position: Maximum distance between inputs to generate
                             unique relation embeddings for. Only relevant
                             when using "dot_product_relative" attention.
      heads_share_relative_embedding: boolean to share relative embeddings
      add_relative_to_values: a boolean for whether to add relative component to
                              values.
      image_shapes: optional tuple of integer scalars.
                    see comments for attention_image_summary()
      block_length: an integer - relevant for "local_mask_right"
      block_width: an integer - relevant for "local_unmasked"
      cache: dict containing Tensors which are the results of previous
             attentions, used for fast decoding. Expects the dict to contrain two
             keys ('k' and 'v'), for the initial call the values for these keys
             should be empty Tensors of the appropriate shape.
                 'k' [batch_size, 0, key_channels]
                 'v' [batch_size, 0, value_channels]
      gap_size: Integer option for dilated attention to indicate spacing between
                memory blocks.
      num_memory_blocks: Integer option to indicate how many memory blocks to look
                         at.
      name: an optional string.
      save_weights_to: an optional dictionary to capture attention weights
        for vizualization; the weights tensor will be appended there under
        a string key created from the variable scope (including name).
      make_image_summary: Whether to make an attention image summary.
      dropout_broadcast_dims:  an optional list of integers less than 4
        specifying in which dimensions to broadcast the dropout decisions.
        saves memory.
      vars_3d: use 3-dimensional variables for input/output transformations
      **kwargs (dict): Parameters for the attention function
    Caching:
      WARNING: For decoder self-attention, i.e. when memory_antecedent == None,
      the caching assumes that the bias contains future masking.
      The caching works by saving all the previous key and value values so that
      you are able to send just the last query location to this attention
      function. I.e. if the cache dict is provided it assumes the query is of the
      shape [batch_size, 1, hidden_dim] rather than the full memory.
    Returns:
      The result of the attention transformation. The output shape is
          [batch_size, length_q, hidden_dim]
      unless the cache dict is provided in which case only the last memory
      position is calculated and the output shape is [batch_size, 1, hidden_dim]
      Optionally returns an additional loss parameters (ex: load balance loss for
      the experts) returned by the attention_type function.
    Raises:
      ValueError: if the key depth or value depth are not divisible by the
        number of attention heads.
    """
    if total_key_depth % num_heads != 0:
        raise ValueError("Key depth (%d) must be divisible by the number of "
                         "attention heads (%d)." % (total_key_depth, num_heads))
    if total_value_depth % num_heads != 0:
        raise ValueError("Value depth (%d) must be divisible by the number of "
                         "attention heads (%d)." % (total_value_depth, num_heads))

    with tf.variable_scope(name, default_name="multihead_attention",
                           values=[query_antecedent, memory_antecedent]):

        if cache is None or memory_antecedent is None:
            if memory_antecedent is None:
                memory_antecedent = query_antecedent
            q, k, v = query_antecedent, memory_antecedent, memory_antecedent
        if cache is not None:
            if attention_type not in ["dot_product", "dot_product_relative"]:
                # TODO(petershaw): Support caching when using relative position
                # representations, i.e. "dot_product_relative" attention.
                raise NotImplementedError(
                    "Caching is not guaranteed to work with attention types other than"
                    " dot_product.")
            if bias is None:
                raise ValueError("Bias required for caching. See function docstring "
                                 "for details.")

            if memory_antecedent is not None:
                # Encoder-Decoder Attention Cache
                q = query_antecedent
                k = cache["k_encdec"]
                v = cache["v_encdec"]
            else:
                k = common_attention.split_heads(k, num_heads)
                v = common_attention.split_heads(v, num_heads)
                decode_loop_step = kwargs.get("decode_loop_step")
                if decode_loop_step is None:
                    k = cache["k"] = tf.concat([cache["k"], k], axis=2)
                    v = cache["v"] = tf.concat([cache["v"], v], axis=2)
                else:
                    # Inplace update is required for inference on TPU.
                    # Inplace_ops only supports inplace_update on the first dimension.
                    # The performance of current implementation is better than updating
                    # the tensor by adding the result of matmul(one_hot,
                    # update_in_current_step)
                    tmp_k = tf.transpose(cache["k"], perm=[2, 0, 1, 3])
                    tmp_k = inplace_ops.alias_inplace_update(
                        tmp_k, decode_loop_step, tf.squeeze(k, axis=2))
                    k = cache["k"] = tf.transpose(tmp_k, perm=[1, 2, 0, 3])
                    tmp_v = tf.transpose(cache["v"], perm=[2, 0, 1, 3])
                    tmp_v = inplace_ops.alias_inplace_update(
                        tmp_v, decode_loop_step, tf.squeeze(v, axis=2))
                    v = cache["v"] = tf.transpose(tmp_v, perm=[1, 2, 0, 3])

        q = common_attention.split_heads(q, num_heads)

        if cache is None:
            k = common_attention.split_heads(k, num_heads)
            v = common_attention.split_heads(v, num_heads)

        key_depth_per_head = total_key_depth // num_heads
        if not vars_3d:
            q *= key_depth_per_head ** -0.5

        additional_returned_value = None
        if callable(attention_type):  # Generic way to extend multihead_attention
            x = attention_type(q, k, v, **kwargs)
            if isinstance(x, tuple):
                x, additional_returned_value = x  # Unpack
        elif attention_type == "dot_product":
            x = common_attention.dot_product_attention(q, k, v, bias, dropout_rate, image_shapes,
                                                       save_weights_to=save_weights_to,
                                                       make_image_summary=make_image_summary,
                                                       dropout_broadcast_dims=dropout_broadcast_dims)
        elif attention_type == "dot_product_relative":
            x = common_attention.dot_product_attention_relative(
                q,
                k,
                v,
                bias,
                max_relative_position,
                dropout_rate,
                image_shapes,
                save_weights_to=save_weights_to,
                make_image_summary=make_image_summary,
                cache=cache is not None)
        elif attention_type == "dot_product_unmasked_relative_v2":
            x = common_attention.dot_product_unmasked_self_attention_relative_v2(
                q,
                k,
                v,
                bias,
                max_relative_position,
                dropout_rate,
                image_shapes,
                make_image_summary=make_image_summary,
                dropout_broadcast_dims=dropout_broadcast_dims,
                heads_share_relative_embedding=heads_share_relative_embedding,
                add_relative_to_values=add_relative_to_values)
        elif attention_type == "dot_product_relative_v2":
            x = common_attention.dot_product_self_attention_relative_v2(
                q,
                k,
                v,
                bias,
                max_relative_position,
                dropout_rate,
                image_shapes,
                make_image_summary=make_image_summary,
                dropout_broadcast_dims=dropout_broadcast_dims,
                heads_share_relative_embedding=heads_share_relative_embedding,
                add_relative_to_values=add_relative_to_values)
        elif attention_type == "local_within_block_mask_right":
            x = common_attention.masked_within_block_local_attention_1d(
                q, k, v, block_length=block_length)
        elif attention_type == "local_relative_mask_right":
            x = common_attention.masked_relative_local_attention_1d(
                q,
                k,
                v,
                block_length=block_length,
                make_image_summary=make_image_summary,
                dropout_rate=dropout_rate,
                heads_share_relative_embedding=heads_share_relative_embedding,
                add_relative_to_values=add_relative_to_values,
                name="masked_relative_local_attention_1d")
        elif attention_type == "local_mask_right":
            x = common_attention.masked_local_attention_1d(
                q,
                k,
                v,
                block_length=block_length,
                make_image_summary=make_image_summary)
        elif attention_type == "local_unmasked":
            x = common_attention.local_attention_1d(
                q, k, v, block_length=block_length, filter_width=block_width)
        elif attention_type == "masked_dilated_1d":
            x = common_attention.masked_dilated_self_attention_1d(q, k, v, block_length, block_width,
                                                                  gap_size, num_memory_blocks)
        else:
            assert attention_type == "unmasked_dilated_1d"
            x = common_attention.dilated_self_attention_1d(q, k, v, block_length, block_width,
                                                           gap_size, num_memory_blocks)
        x = common_attention.combine_heads(x)

        # Set last dim specifically.
        x.set_shape(x.shape.as_list()[:-1] + [total_value_depth])

        if vars_3d:
            o_var = tf.get_variable(
                "o", [num_heads, total_value_depth // num_heads, output_depth])
            o_var = tf.cast(o_var, x.dtype)
            o_var = tf.reshape(o_var, [total_value_depth, output_depth])
            x = tf.tensordot(x, o_var, axes=1)
        else:
            x = common_layers.dense(
                x, output_depth, use_bias=False, name="output_transform")
        if additional_returned_value is not None:
            return x, additional_returned_value
        return x


@registry.register_model
class SimplifiedTransformer(t2t_model.T2TModel):
    """Attention net.  See file docstring."""

    def __init__(self, *args, **kwargs):
        super(SimplifiedTransformer, self).__init__(*args, **kwargs)
        self.attention_weights = {}  # For visualizing attention heads.

    def encode(self, inputs, target_space, hparams, features=None, losses=None):
        """Encode transformer inputs.

        Args:
          inputs: Transformer inputs [batch_size, input_length, 1, hidden_dim] which
            will be flattened along the two spatial dimensions.
          target_space: scalar, target space ID.
          hparams: hyperparameters for model.
          features: optionally pass the entire features dictionary as well.
            This is needed now for "packed" datasets.
          losses: optional list onto which to append extra training losses

        Returns:
          Tuple of:
              encoder_output: Encoder representation.
                  [batch_size, input_length, hidden_dim]
              encoder_decoder_attention_bias: Bias and mask weights for
                  encoder-decoder attention. [batch_size, input_length]
        """
        inputs = common_layers.flatten4d3d(inputs)

        encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
            transformer_prepare_encoder(
                inputs, target_space, hparams, features=features))

        mlperf_log.transformer_print(
            key=mlperf_log.MODEL_HP_LAYER_POSTPROCESS_DROPOUT,
            value=hparams.layer_prepostprocess_dropout,
            hparams=hparams)

        encoder_input = tf.nn.dropout(encoder_input,
                                      1.0 - hparams.layer_prepostprocess_dropout)

        attn_bias_for_padding = None
        # Otherwise the encoder will just use encoder_self_attention_bias.
        if hparams.unidirectional_encoder:
            attn_bias_for_padding = encoder_decoder_attention_bias

        encoder_output = transformer_encoder(
            encoder_input,
            self_attention_bias,
            hparams,
            nonpadding=features_to_nonpadding(features, "inputs"),
            save_weights_to=self.attention_weights,
            make_image_summary=not common_layers.is_xla_compiled(),
            losses=losses,
            attn_bias_for_padding=attn_bias_for_padding)

        return encoder_output, encoder_decoder_attention_bias

    def decode(self,
               decoder_input,
               encoder_output,
               encoder_decoder_attention_bias,
               decoder_self_attention_bias,
               hparams,
               cache=None,
               decode_loop_step=None,
               nonpadding=None,
               losses=None):
        """Decode Transformer outputs from encoder representation.

        Args:
          decoder_input: inputs to bottom of the model.
              [batch_size, decoder_length, hidden_dim]
          encoder_output: Encoder representation.
              [batch_size, input_length, hidden_dim]
          encoder_decoder_attention_bias: Bias and mask weights for
              encoder-decoder attention. [batch_size, input_length]
          decoder_self_attention_bias: Bias and mask weights for decoder
              self-attention. [batch_size, decoder_length]
          hparams: hyperparameters for model.
          cache: dict, containing tensors which are the results of previous
              attentions, used for fast decoding.
          decode_loop_step: An integer, step number of the decoding loop.
              Only used for inference on TPU.
          nonpadding: optional Tensor with shape [batch_size, decoder_length]
          losses: optional list onto which to append extra training losses

        Returns:
          Final decoder representation. [batch_size, decoder_length, hidden_dim]
        """
        mlperf_log.transformer_print(
            key=mlperf_log.MODEL_HP_LAYER_POSTPROCESS_DROPOUT,
            value=hparams.layer_prepostprocess_dropout,
            hparams=hparams)
        decoder_input = tf.nn.dropout(decoder_input,
                                      1.0 - hparams.layer_prepostprocess_dropout)

        decoder_output = transformer_decoder(
            decoder_input,
            encoder_output,
            decoder_self_attention_bias,
            encoder_decoder_attention_bias,
            hparams,
            cache=cache,
            decode_loop_step=decode_loop_step,
            nonpadding=nonpadding,
            save_weights_to=self.attention_weights,
            losses=losses)

        if (common_layers.is_xla_compiled() and
                hparams.mode == tf.estimator.ModeKeys.TRAIN):
            # TPU does not react kindly to extra dimensions.
            # TODO(noam): remove this once TPU is more forgiving of extra dims.
            return decoder_output
        else:
            # Expand since t2t expects 4d tensors.
            return tf.expand_dims(decoder_output, axis=2)

    def body(self, features):
        """Transformer main model_fn.

        Args:
          features: Map of features to the model. Should contain the following:
              "inputs": Transformer inputs.
                  [batch_size, input_length, 1, hidden_dim].
              "targets": Target decoder outputs.
                  [batch_size, decoder_length, 1, hidden_dim]
              "target_space_id": A scalar int from data_generators.problem.SpaceID.

        Returns:
          Final decoder representation. [batch_size, decoder_length, hidden_dim]
        """
        hparams = self._hparams

        losses = []

        if self.has_input:
            inputs = features["inputs"]
            target_space = features["target_space_id"]
            encoder_output, encoder_decoder_attention_bias = self.encode(
                inputs, target_space, hparams, features=features, losses=losses)
        else:
            encoder_output, encoder_decoder_attention_bias = (None, None)

        targets = features["targets"]
        targets_shape = common_layers.shape_list(targets)
        targets = common_layers.flatten4d3d(targets)
        decoder_input, decoder_self_attention_bias = transformer_prepare_decoder(
            targets, hparams, features=features)
        decoder_output = self.decode(
            decoder_input,
            encoder_output,
            encoder_decoder_attention_bias,
            decoder_self_attention_bias,
            hparams,
            nonpadding=features_to_nonpadding(features, "targets"),
            losses=losses)

        expected_attentions = features.get("expected_attentions")
        if expected_attentions is not None:
            attention_loss = common_attention.encoder_decoder_attention_loss(
                expected_attentions, self.attention_weights,
                hparams.expected_attention_loss_type,
                hparams.expected_attention_loss_multiplier)
            return decoder_output, {"attention_loss": attention_loss}

        ret = tf.reshape(decoder_output, targets_shape)
        if losses:
            return ret, {"extra_loss": tf.add_n(losses)}
        else:
            return ret

    def _greedy_infer(self, features, decode_length, use_tpu=False):
        """Fast version of greedy decoding.

        Args:
          features: an map of string to `Tensor`
          decode_length: an integer.  How many additional timesteps to decode.
          use_tpu: A bool. Whether to build the inference graph for TPU.

        Returns:
          A dict of decoding results {
              "outputs": integer `Tensor` of decoded ids of shape
                  [batch_size, <= decode_length] if beam_size == 1 or
                  [batch_size, top_beams, <= decode_length]
              "scores": decoding log probs from the beam search,
                  None if using greedy decoding (beam_size=1)
          }

        Raises:
          NotImplementedError: If there are multiple data shards.
        """
        # For real-valued modalities use the slow decode path for now.
        if (self._target_modality_is_real or
                self._hparams.self_attention_type != "dot_product"):
            return super(SimplifiedTransformer, self)._greedy_infer(features, decode_length)
        with tf.variable_scope(self.name):
            if use_tpu:
                return self._fast_decode_tpu(features, decode_length)
            return self._fast_decode(features, decode_length)

    def _beam_decode(self,
                     features,
                     decode_length,
                     beam_size,
                     top_beams,
                     alpha,
                     use_tpu=False):
        """Beam search decoding.

        Args:
          features: an map of string to `Tensor`
          decode_length: an integer.  How many additional timesteps to decode.
          beam_size: number of beams.
          top_beams: an integer. How many of the beams to return.
          alpha: Float that controls the length penalty. larger the alpha, stronger
            the preference for longer translations.
          use_tpu: A bool, whether to do beam decode on TPU.

        Returns:
          A dict of decoding results {
              "outputs": integer `Tensor` of decoded ids of shape
                  [batch_size, <= decode_length] if beam_size == 1 or
                  [batch_size, top_beams, <= decode_length]
              "scores": decoding log probs from the beam search,
                  None if using greedy decoding (beam_size=1)
          }
        """
        if (self._hparams.self_attention_type not in ["dot_product",
                                                      "dot_product_relative"]):
            # Caching is not guaranteed to work with attention types other than
            # dot_product.
            # TODO(petershaw): Support fast decoding when using relative
            # position representations, i.e. "dot_product_relative" attention.
            return self._beam_decode_slow(features, decode_length, beam_size,
                                          top_beams, alpha, use_tpu)
        with tf.variable_scope(self.name):
            if use_tpu:
                return self._fast_decode_tpu(
                    features, decode_length, beam_size, top_beams, alpha)
            return self._fast_decode(
                features, decode_length, beam_size, top_beams, alpha)

    def _fast_decode_tpu(self,
                         features,
                         decode_length,
                         beam_size=1,
                         top_beams=1,
                         alpha=1.0):
        """Fast decoding.

        Implements both greedy and beam search decoding on TPU, uses beam search
        iff beam_size > 1, otherwise beam search related arguments are ignored.

        Args:
          features: A map of string to model features.
          decode_length: An integer, how many additional timesteps to decode.
          beam_size: An integer, number of beams.
          top_beams: An integer, how many of the beams to return.
          alpha: A float that controls the length penalty. Larger the alpha,
            stronger the preference for longer translations.

        Returns:
          A dict of decoding results {
              "outputs": integer `Tensor` of decoded ids of shape
                  [batch_size, <= decode_length] if beam_size == 1 or
                  [batch_size, top_beams, <= decode_length]
              "scores": decoding log probs from the beam search,
                  None if using greedy decoding (beam_size=1)
          }.

        Raises:
          NotImplementedError: If there are multiple data shards.
        """
        if self._num_datashards != 1:
            raise NotImplementedError("Fast decoding only supports a single shard.")
        if "targets_segmentation" in features:
            raise NotImplementedError(
                "Decoding not supported on packed datasets "
                " If you want to decode from a dataset, use the non-packed version"
                " of the dataset when decoding.")
        dp = self._data_parallelism
        hparams = self._hparams
        target_modality = self._problem_hparams.modality["targets"]

        if self.has_input:
            inputs = features["inputs"]
            if target_modality.is_class_modality:
                decode_length = 1
            else:
                decode_length = (
                        common_layers.shape_list(inputs)[1] + features.get(
                    "decode_length", decode_length))

            # TODO(llion): Clean up this reshaping logic.
            inputs = tf.expand_dims(inputs, axis=1)
            if len(inputs.shape) < 5:
                inputs = tf.expand_dims(inputs, axis=4)
            s = common_layers.shape_list(inputs)
            batch_size = s[0]
            inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
            # _shard_features called to ensure that the variable names match
            inputs = self._shard_features({"inputs": inputs})["inputs"]
            input_modality = self._problem_hparams.modality["inputs"]
            with tf.variable_scope(input_modality.name):
                inputs = input_modality.bottom_sharded(inputs, dp)
            with tf.variable_scope("body"):
                encoder_output, encoder_decoder_attention_bias = dp(
                    self.encode,
                    inputs,
                    features["target_space_id"],
                    hparams,
                    features=features)
            encoder_output = encoder_output[0]
            encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
            partial_targets = None
        else:
            # The problem has no inputs.
            encoder_output = None
            encoder_decoder_attention_bias = None

            # Prepare partial targets.
            # In either features["inputs"] or features["targets"].
            # We force the outputs to begin with these sequences.
            partial_targets = features.get("inputs")
            if partial_targets is None:
                partial_targets = features["targets"]
            assert partial_targets is not None
            partial_targets = common_layers.expand_squeeze_to_nd(partial_targets, 2)
            partial_targets = tf.to_int64(partial_targets)
            partial_targets_shape = common_layers.shape_list(partial_targets)
            partial_targets_length = partial_targets_shape[1]
            decode_length = (
                    partial_targets_length + features.get("decode_length", decode_length))
            batch_size = partial_targets_shape[0]

        if hparams.pos == "timing":
            positional_encoding = common_attention.get_timing_signal_1d(
                decode_length + 1, hparams.hidden_size)
        elif hparams.pos == "emb":
            positional_encoding = common_attention.add_positional_embedding(
                tf.zeros([1, decode_length + 1, hparams.hidden_size]),
                hparams.max_length, "body/targets_positional_embedding", None)
        else:
            positional_encoding = None

        def preprocess_targets(targets, i):
            """Performs preprocessing steps on the targets to prepare for the decoder.

            This includes:
              - Embedding the ids.
              - Flattening to 3D tensor.
              - Optionally adding timing signals.

            Args:
              targets: A tensor, inputs ids to the decoder. [batch_size, 1].
              i: An integer, Step number of the decoding loop.

            Returns:
              A tensor, processed targets [batch_size, 1, hidden_dim].
            """
            # _shard_features called to ensure that the variable names match
            targets = self._shard_features({"targets": targets})["targets"]
            with tf.variable_scope(target_modality.name):
                targets = target_modality.targets_bottom_sharded(targets, dp)[0]
            targets = common_layers.flatten4d3d(targets)

            # TODO(llion): Explain! Is this even needed?
            targets = tf.cond(
                tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

            if positional_encoding is not None:
                positional_encoding_shape = positional_encoding.shape.as_list()
                targets += tf.slice(
                    positional_encoding, [0, i, 0],
                    [positional_encoding_shape[0], 1, positional_encoding_shape[2]])
            return targets

        decoder_self_attention_bias = (
            common_attention.attention_bias_lower_triangle(decode_length))
        if hparams.proximity_bias:
            decoder_self_attention_bias += common_attention.attention_bias_proximal(
                decode_length)

        def symbols_to_logits_tpu_fn(ids, i, cache):
            """Go from ids to logits for next symbol on TPU.

            Args:
              ids: A tensor, symbol IDs.
              i: An integer, step number of the decoding loop. Only used for inference
                  on TPU.
              cache: A dict, containing tensors which are the results of previous
                  attentions, used for fast decoding.

            Returns:
              ret: A tensor, computed logits.
              cache: A dict, containing tensors which are the results of previous
                  attentions, used for fast decoding.
            """
            ids = ids[:, -1:]
            targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
            targets = preprocess_targets(targets, i)

            bias_shape = decoder_self_attention_bias.shape.as_list()
            bias = tf.slice(decoder_self_attention_bias, [0, 0, i, 0],
                            [bias_shape[0], bias_shape[1], 1, bias_shape[3]])

            with tf.variable_scope("body"):
                body_outputs = dp(
                    self.decode,
                    targets,
                    cache.get("encoder_output"),
                    cache.get("encoder_decoder_attention_bias"),
                    bias,
                    hparams,
                    cache,
                    i,
                    nonpadding=features_to_nonpadding(features, "targets"))

            with tf.variable_scope(target_modality.name):
                logits = target_modality.top_sharded(body_outputs, None, dp)[0]

            ret = tf.squeeze(logits, axis=[1, 2, 3])
            if partial_targets is not None:
                # If the position is within the given partial targets, we alter the
                # logits to always return those values.
                # A faster approach would be to process the partial targets in one
                # iteration in order to fill the corresponding parts of the cache.
                # This would require broader changes, though.
                vocab_size = tf.shape(ret)[1]

                def forced_logits():
                    return tf.one_hot(
                        tf.tile(
                            tf.slice(partial_targets, [0, i],
                                     [partial_targets.shape.as_list()[0], 1]),
                            [beam_size]), vocab_size, 0.0, -1e9)

                ret = tf.cond(
                    tf.less(i, partial_targets_length), forced_logits, lambda: ret)
            return ret, cache

        ret = fast_decode_tpu(
            encoder_output=encoder_output,
            encoder_decoder_attention_bias=encoder_decoder_attention_bias,
            symbols_to_logits_fn=symbols_to_logits_tpu_fn,
            hparams=hparams,
            decode_length=decode_length,
            vocab_size=target_modality.top_dimensionality,
            beam_size=beam_size,
            top_beams=top_beams,
            alpha=alpha,
            batch_size=batch_size,
            force_decode_length=self._decode_hparams.force_decode_length)
        if partial_targets is not None:
            if beam_size <= 1 or top_beams <= 1:
                ret["outputs"] = ret["outputs"][:, partial_targets_length:]
            else:
                ret["outputs"] = ret["outputs"][:, :, partial_targets_length:]
        return ret

    def _fast_decode(self,
                     features,
                     decode_length,
                     beam_size=1,
                     top_beams=1,
                     alpha=1.0):
        """Fast decoding.

        Implements both greedy and beam search decoding, uses beam search iff
        beam_size > 1, otherwise beam search related arguments are ignored.

        Args:
          features: a map of string to model  features.
          decode_length: an integer.  How many additional timesteps to decode.
          beam_size: number of beams.
          top_beams: an integer. How many of the beams to return.
          alpha: Float that controls the length penalty. larger the alpha, stronger
            the preference for longer translations.

        Returns:
          A dict of decoding results {
              "outputs": integer `Tensor` of decoded ids of shape
                  [batch_size, <= decode_length] if beam_size == 1 or
                  [batch_size, top_beams, <= decode_length]
              "scores": decoding log probs from the beam search,
                  None if using greedy decoding (beam_size=1)
          }

        Raises:
          NotImplementedError: If there are multiple data shards.
        """
        if self._num_datashards != 1:
            raise NotImplementedError("Fast decoding only supports a single shard.")
        dp = self._data_parallelism
        hparams = self._hparams
        target_modality = self._problem_hparams.modality["targets"]
        if "targets_segmentation" in features:
            raise NotImplementedError(
                "Decoding not supported on packed datasets "
                " If you want to decode from a dataset, use the non-packed version"
                " of the dataset when decoding.")
        if self.has_input:
            inputs = features["inputs"]
            if target_modality.is_class_modality:
                decode_length = 1
            else:
                decode_length = (
                        common_layers.shape_list(inputs)[1] + features.get(
                    "decode_length", decode_length))

            # TODO(llion): Clean up this reshaping logic.
            inputs = tf.expand_dims(inputs, axis=1)
            if len(inputs.shape) < 5:
                inputs = tf.expand_dims(inputs, axis=4)
            s = common_layers.shape_list(inputs)
            batch_size = s[0]
            inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
            # _shard_features called to ensure that the variable names match
            inputs = self._shard_features({"inputs": inputs})["inputs"]
            input_modality = self._problem_hparams.modality["inputs"]
            with tf.variable_scope(input_modality.name):
                inputs = input_modality.bottom_sharded(inputs, dp)
            with tf.variable_scope("body"):
                encoder_output, encoder_decoder_attention_bias = dp(
                    self.encode,
                    inputs,
                    features["target_space_id"],
                    hparams,
                    features=features)
            encoder_output = encoder_output[0]
            encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
            partial_targets = None
        else:
            # The problem has no inputs.
            encoder_output = None
            encoder_decoder_attention_bias = None

            # Prepare partial targets.
            # In either features["inputs"] or features["targets"].
            # We force the outputs to begin with these sequences.
            partial_targets = features.get("inputs")
            if partial_targets is None:
                partial_targets = features["targets"]
            assert partial_targets is not None
            partial_targets = common_layers.expand_squeeze_to_nd(partial_targets, 2)
            partial_targets = tf.to_int64(partial_targets)
            partial_targets_shape = common_layers.shape_list(partial_targets)
            partial_targets_length = partial_targets_shape[1]
            decode_length = (
                    partial_targets_length + features.get("decode_length", decode_length))
            batch_size = partial_targets_shape[0]

        if hparams.pos == "timing":
            positional_encoding = common_attention.get_timing_signal_1d(
                decode_length + 1, hparams.hidden_size)
        elif hparams.pos == "emb":
            positional_encoding = common_attention.add_positional_embedding(
                tf.zeros([1, decode_length, hparams.hidden_size]),
                hparams.max_length, "body/targets_positional_embedding", None)
        else:
            positional_encoding = None

        def preprocess_targets(targets, i):
            """Performs preprocessing steps on the targets to prepare for the decoder.

            This includes:
              - Embedding the ids.
              - Flattening to 3D tensor.
              - Optionally adding timing signals.

            Args:
              targets: inputs ids to the decoder. [batch_size, 1]
              i: scalar, Step number of the decoding loop.

            Returns:
              Processed targets [batch_size, 1, hidden_dim]
            """
            # _shard_features called to ensure that the variable names match
            targets = self._shard_features({"targets": targets})["targets"]
            with tf.variable_scope(target_modality.name):
                targets = target_modality.targets_bottom_sharded(targets, dp)[0]
            targets = common_layers.flatten4d3d(targets)

            # TODO(llion): Explain! Is this even needed?
            targets = tf.cond(
                tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

            if positional_encoding is not None:
                targets += positional_encoding[:, i:i + 1]
            return targets

        decoder_self_attention_bias = (
            common_attention.attention_bias_lower_triangle(decode_length))
        if hparams.proximity_bias:
            decoder_self_attention_bias += common_attention.attention_bias_proximal(
                decode_length)

        def symbols_to_logits_fn(ids, i, cache):
            """Go from ids to logits for next symbol."""
            ids = ids[:, -1:]
            targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
            targets = preprocess_targets(targets, i)

            bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

            with tf.variable_scope("body"):
                body_outputs = dp(
                    self.decode,
                    targets,
                    cache.get("encoder_output"),
                    cache.get("encoder_decoder_attention_bias"),
                    bias,
                    hparams,
                    cache,
                    nonpadding=features_to_nonpadding(features, "targets"))

            with tf.variable_scope(target_modality.name):
                logits = target_modality.top_sharded(body_outputs, None, dp)[0]

            ret = tf.squeeze(logits, axis=[1, 2, 3])
            if partial_targets is not None:
                # If the position is within the given partial targets, we alter the
                # logits to always return those values.
                # A faster approach would be to process the partial targets in one
                # iteration in order to fill the corresponding parts of the cache.
                # This would require broader changes, though.
                vocab_size = tf.shape(ret)[1]

                def forced_logits():
                    return tf.one_hot(
                        tf.tile(partial_targets[:, i], [beam_size]), vocab_size, 0.0,
                        -1e9)

                ret = tf.cond(
                    tf.less(i, partial_targets_length), forced_logits, lambda: ret)
            return ret, cache

        ret = fast_decode(
            encoder_output=encoder_output,
            encoder_decoder_attention_bias=encoder_decoder_attention_bias,
            symbols_to_logits_fn=symbols_to_logits_fn,
            hparams=hparams,
            decode_length=decode_length,
            vocab_size=target_modality.top_dimensionality,
            beam_size=beam_size,
            top_beams=top_beams,
            alpha=alpha,
            batch_size=batch_size,
            force_decode_length=self._decode_hparams.force_decode_length)
        if partial_targets is not None:
            if beam_size <= 1 or top_beams <= 1:
                ret["outputs"] = ret["outputs"][:, partial_targets_length:]
            else:
                ret["outputs"] = ret["outputs"][:, :, partial_targets_length:]
        return ret


def fast_decode_tpu(encoder_output,
                    encoder_decoder_attention_bias,
                    symbols_to_logits_fn,
                    hparams,
                    decode_length,
                    vocab_size,
                    beam_size=1,
                    top_beams=1,
                    alpha=1.0,
                    sos_id=0,
                    eos_id=beam_search.EOS_ID,
                    batch_size=None,
                    force_decode_length=False,
                    scope_prefix="body/"):
    """Given encoder output and a symbols to logits function, does fast decoding.

    Implements both greedy and beam search decoding for TPU, uses beam search iff
    beam_size > 1, otherwise beam search related arguments are ignored.

    Args:
      encoder_output: A tensor, output from encoder.
      encoder_decoder_attention_bias: A tensor, bias for use in encoder-decoder
          attention.
      symbols_to_logits_fn: Incremental decoding, function mapping triple
          `(ids, step, cache)` to symbol logits.
      hparams: Run hyperparameters.
      decode_length: An integer, how many additional timesteps to decode.
      vocab_size: Output vocabulary size.
      beam_size: An integer, number of beams.
      top_beams: An integer, how many of the beams to return.
      alpha: A float that controls the length penalty. Larger the alpha, stronger
        the preference for longer translations.
      sos_id: Start-of-sequence symbol.
      eos_id: End-of-sequence symbol.
      batch_size: An integer, must be passed if there is no input.
      force_decode_length: A bool, whether to force the full decode length, or if
          False, stop when all beams hit eos_id.
      scope_prefix: str, prefix for decoder layer variable scopes.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if top_beams == 1 or
              [batch_size, top_beams, <= decode_length] otherwise
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }.

    Raises:
      NotImplementedError: If beam size > 1 with partial targets.
    """
    if encoder_output is not None:
        batch_size = common_layers.shape_list(encoder_output)[0]

    key_channels = hparams.attention_key_channels or hparams.hidden_size
    value_channels = hparams.attention_value_channels or hparams.hidden_size
    num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers
    vars_3d_num_heads = (
        hparams.num_heads if hparams.get("attention_variables_3d") else 0)

    cache = {
        "layer_%d" % layer: {
            "k":
                common_attention.split_heads(
                    tf.zeros([batch_size, decode_length, key_channels]),
                    hparams.num_heads),
            "v":
                common_attention.split_heads(
                    tf.zeros([batch_size, decode_length, value_channels]),
                    hparams.num_heads),
        } for layer in range(num_layers)
    }

    # If `ffn_layer` is in `["dense_relu_dense" or "conv_hidden_relu"]`, then the
    # cache key "f" won't be used, which means that the` shape of cache["f"]`
    # won't be changed to
    # `[beamsize*batch_size, decode_length, hparams.hidden_size]` and may cause
    # error when applying `nest.map reshape function` on it.
    if hparams.ffn_layer not in ["dense_relu_dense", "conv_hidden_relu"]:
        for layer in range(num_layers):
            cache["layer_%d" % layer]["f"] = tf.zeros(
                [batch_size, 0, hparams.hidden_size])

    if encoder_output is not None:
        for layer in range(num_layers):
            layer_name = "layer_%d" % layer
            with tf.variable_scope(
                    "%sdecoder/%s/encdec_attention/multihead_attention" % (scope_prefix,
                                                                           layer_name)):
                k_encdec = common_attention.compute_attention_component(
                    encoder_output, key_channels, name="k",
                    vars_3d_num_heads=vars_3d_num_heads)
                k_encdec = common_attention.split_heads(k_encdec, hparams.num_heads)
                v_encdec = common_attention.compute_attention_component(
                    encoder_output, value_channels, name="v",
                    vars_3d_num_heads=vars_3d_num_heads)
                v_encdec = common_attention.split_heads(v_encdec, hparams.num_heads)
            cache[layer_name]["k_encdec"] = k_encdec
            cache[layer_name]["v_encdec"] = v_encdec

        cache["encoder_output"] = encoder_output
        cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_SEQ_BEAM_SEARCH,
        value={
            "vocab_size": vocab_size,
            "batch_size": batch_size,
            "beam_size": beam_size,
            "alpha": alpha,
            "max_decode_length": decode_length
        },
        hparams=hparams)
    if beam_size > 1:  # Beam Search
        initial_ids = sos_id * tf.ones([batch_size], dtype=tf.int32)
        decoded_ids, scores, _ = beam_search.beam_search(
            symbols_to_logits_fn,
            initial_ids,
            beam_size,
            decode_length,
            vocab_size,
            alpha,
            states=cache,
            eos_id=eos_id,
            stop_early=(top_beams == 1),
            use_tpu=True)

        if top_beams == 1:
            decoded_ids = decoded_ids[:, 0, 1:]
            scores = scores[:, 0]
        else:
            decoded_ids = decoded_ids[:, :top_beams, 1:]
            scores = scores[:, :top_beams]
    else:  # Greedy
        def inner_loop(i, hit_eos, next_id, decoded_ids, cache, log_prob):
            """One step of greedy decoding."""
            logits, cache = symbols_to_logits_fn(next_id, i, cache)
            log_probs = common_layers.log_prob_from_logits(logits)
            temperature = getattr(hparams, "sampling_temp", 0.0)
            if hparams.sampling_method == "argmax":
                temperature = 0.0
            next_id = common_layers.sample_with_temperature(logits, temperature)
            hit_eos |= tf.equal(next_id, eos_id)

            log_prob_indices = tf.stack(
                [tf.range(tf.to_int64(batch_size)), next_id], axis=1)
            log_prob += tf.gather_nd(log_probs, log_prob_indices)

            next_id = tf.expand_dims(next_id, axis=1)
            decoded_ids = tf.transpose(decoded_ids)
            decoded_ids = inplace_ops.alias_inplace_update(
                decoded_ids, i, tf.squeeze(next_id, axis=1))
            decoded_ids = tf.transpose(decoded_ids)
            return i + 1, hit_eos, next_id, decoded_ids, cache, log_prob

        def is_not_finished(i, hit_eos, *_):
            finished = i >= decode_length
            if not force_decode_length:
                finished |= tf.reduce_all(hit_eos)
            return tf.logical_not(finished)

        decoded_ids = tf.zeros([batch_size, decode_length], dtype=tf.int64)
        hit_eos = tf.fill([batch_size], False)
        next_id = sos_id * tf.ones([batch_size, 1], dtype=tf.int64)
        initial_log_prob = tf.zeros([batch_size], dtype=tf.float32)

        def compute_cache_shape_invariants(tensor):
            return tf.TensorShape(tensor.shape.as_list())

        _, _, _, decoded_ids, _, log_prob = tf.while_loop(
            is_not_finished,
            inner_loop, [
                tf.constant(0), hit_eos, next_id, decoded_ids, cache,
                initial_log_prob
            ],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([batch_size]),
                tf.TensorShape([batch_size, 1]),
                tf.TensorShape([batch_size, decode_length]),
                nest.map_structure(compute_cache_shape_invariants, cache),
                tf.TensorShape([batch_size]),
            ])
        scores = log_prob

    return {"outputs": decoded_ids, "scores": scores}


def fast_decode(encoder_output,
                encoder_decoder_attention_bias,
                symbols_to_logits_fn,
                hparams,
                decode_length,
                vocab_size,
                beam_size=1,
                top_beams=1,
                alpha=1.0,
                sos_id=0,
                eos_id=beam_search.EOS_ID,
                batch_size=None,
                force_decode_length=False,
                scope_prefix="body/",
                cache=None):
    """Given encoder output and a symbols to logits function, does fast decoding.

    Implements both greedy and beam search decoding, uses beam search iff
    beam_size > 1, otherwise beam search related arguments are ignored.

    Args:
      encoder_output: Output from encoder.
      encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
        attention
      symbols_to_logits_fn: Incremental decoding; function mapping triple
        `(ids, step, cache)` to symbol logits.
      hparams: run hyperparameters
      decode_length: an integer.  How many additional timesteps to decode.
      vocab_size: Output vocabulary size.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.
      sos_id: End-of-sequence symbol in beam search.
      eos_id: End-of-sequence symbol in beam search.
      batch_size: an integer scalar - must be passed if there is no input
      force_decode_length: bool, whether to force the full decode length, or if
        False, stop when all beams hit eos_id.
      scope_prefix: str, prefix for decoder layer variable scopes.
      cache: cache dictionary for additional predictions.

    Returns:
        A dict of decoding results {
            "outputs": integer `Tensor` of decoded ids of shape
                [batch_size, <= decode_length] if top_beams == 1 or
                [batch_size, top_beams, <= decode_length] otherwise
            "scores": decoding log probs from the beam search,
                None if using greedy decoding (beam_size=1)
        }

      Raises:
        NotImplementedError: If beam size > 1 with partial targets.
    """
    if encoder_output is not None:
        batch_size = common_layers.shape_list(encoder_output)[0]

    key_channels = hparams.attention_key_channels or hparams.hidden_size
    value_channels = hparams.attention_value_channels or hparams.hidden_size
    num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers
    vars_3d_num_heads = (
        hparams.num_heads if hparams.get("attention_variables_3d") else 0)

    if cache is None:
        cache = {}
    cache.update({
        "layer_%d" % layer: {
            "k":
                common_attention.split_heads(
                    tf.zeros([batch_size, 0, key_channels]), hparams.num_heads),
            "v":
                common_attention.split_heads(
                    tf.zeros([batch_size, 0, value_channels]), hparams.num_heads),
        } for layer in range(num_layers)
    })

    # If `ffn_layer` is in `["dense_relu_dense" or "conv_hidden_relu"]`, then the
    # cache key "f" won't be used, which means that the` shape of cache["f"]`
    # won't be changed to
    # `[beamsize*batch_size, decode_length, hparams.hidden_size]` and may cause
    # error when applying `nest.map reshape function` on it.
    if hparams.ffn_layer not in ["dense_relu_dense", "conv_hidden_relu"]:
        for layer in range(num_layers):
            cache["layer_%d" % layer]["f"] = tf.zeros(
                [batch_size, 0, hparams.hidden_size])

    if encoder_output is not None:
        for layer in range(num_layers):
            layer_name = "layer_%d" % layer
            with tf.variable_scope(
                    "%sdecoder/%s/encdec_attention/multihead_attention" % (scope_prefix,
                                                                           layer_name)):
                k_encdec = common_attention.compute_attention_component(
                    encoder_output, key_channels, name="k",
                    vars_3d_num_heads=vars_3d_num_heads)
                k_encdec = common_attention.split_heads(k_encdec, hparams.num_heads)
                v_encdec = common_attention.compute_attention_component(
                    encoder_output, value_channels, name="v",
                    vars_3d_num_heads=vars_3d_num_heads)
                v_encdec = common_attention.split_heads(v_encdec, hparams.num_heads)
            cache[layer_name]["k_encdec"] = k_encdec
            cache[layer_name]["v_encdec"] = v_encdec

        cache["encoder_output"] = encoder_output
        cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

    if beam_size > 1:  # Beam Search
        initial_ids = sos_id * tf.ones([batch_size], dtype=tf.int32)
        decoded_ids, scores, cache = beam_search.beam_search(
            symbols_to_logits_fn,
            initial_ids,
            beam_size,
            decode_length,
            vocab_size,
            alpha,
            states=cache,
            eos_id=eos_id,
            stop_early=(top_beams == 1))

        if top_beams == 1:
            decoded_ids = decoded_ids[:, 0, 1:]
            scores = scores[:, 0]
        else:
            decoded_ids = decoded_ids[:, :top_beams, 1:]
            scores = scores[:, :top_beams]
    else:  # Greedy

        def inner_loop(i, hit_eos, next_id, decoded_ids, cache, log_prob):
            """One step of greedy decoding."""
            logits, cache = symbols_to_logits_fn(next_id, i, cache)
            log_probs = common_layers.log_prob_from_logits(logits)
            temperature = getattr(hparams, "sampling_temp", 0.0)
            if hparams.sampling_method == "argmax":
                temperature = 0.0
            next_id = common_layers.sample_with_temperature(logits, temperature)
            hit_eos |= tf.equal(next_id, eos_id)

            log_prob_indices = tf.stack(
                [tf.range(tf.to_int64(batch_size)), next_id], axis=1)
            log_prob += tf.gather_nd(log_probs, log_prob_indices)

            next_id = tf.expand_dims(next_id, axis=1)
            decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
            return i + 1, hit_eos, next_id, decoded_ids, cache, log_prob

        def is_not_finished(i, hit_eos, *_):
            finished = i >= decode_length
            if not force_decode_length:
                finished |= tf.reduce_all(hit_eos)
            return tf.logical_not(finished)

        decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int64)
        hit_eos = tf.fill([batch_size], False)
        next_id = sos_id * tf.ones([batch_size, 1], dtype=tf.int64)
        initial_log_prob = tf.zeros([batch_size], dtype=tf.float32)
        _, _, _, decoded_ids, cache, log_prob = tf.while_loop(
            is_not_finished,
            inner_loop, [
                tf.constant(0), hit_eos, next_id, decoded_ids, cache,
                initial_log_prob
            ],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None]),
                nest.map_structure(beam_search.get_state_shape_invariants, cache),
                tf.TensorShape([None]),
            ])
        scores = log_prob

    return {"outputs": decoded_ids, "scores": scores, "cache": cache}


@registry.register_model
class SimplifiedTransformerScorer(SimplifiedTransformer):
    """Transformer model, but only scores in PREDICT mode.

    Checkpoints between Transformer and TransformerScorer are interchangeable.
    """

    def __init__(self, *args, **kwargs):
        super(SimplifiedTransformerScorer, self).__init__(*args, **kwargs)
        self._name = "transformer"
        self._base_name = "transformer"

    def infer(self,
              features=None,
              decode_length=50,
              beam_size=1,
              top_beams=1,
              alpha=0.0,
              use_tpu=False):
        """Returns the targets and their log probabilities."""
        del decode_length, beam_size, top_beams, alpha, use_tpu
        assert features is not None

        # Run the model
        self.hparams.force_full_predict = True
        with tf.variable_scope(self.name):
            logits, _ = self.model_fn(features)
        assert len(logits.shape) == 5  # [batch, time, 1, 1, vocab]
        logits = tf.squeeze(logits, [2, 3])

        # Compute the log probabilities
        log_probs = common_layers.log_prob_from_logits(logits)

        targets = features["targets"]
        assert len(targets.shape) == 4  # [batch, time, 1, 1]
        targets = tf.squeeze(targets, [2, 3])

        # Slice out the log_probs of the targets
        log_probs = common_layers.index_last_dim_with_indices(log_probs, targets)

        # Sum over time to get the log_prob of the sequence
        scores = tf.reduce_sum(log_probs, axis=1)

        return {"outputs": targets, "scores": scores}


@registry.register_model
class SimplifiedTransformerEncoder(t2t_model.T2TModel):
    """Transformer, encoder only."""

    def body(self, features):
        hparams = self._hparams
        inputs = features["inputs"]
        target_space = features["target_space_id"]

        inputs = common_layers.flatten4d3d(inputs)

        (encoder_input, encoder_self_attention_bias, _) = (
            transformer_prepare_encoder(inputs, target_space, hparams))

        encoder_input = tf.nn.dropout(encoder_input,
                                      1.0 - hparams.layer_prepostprocess_dropout)
        encoder_output = transformer_encoder(
            encoder_input,
            encoder_self_attention_bias,
            hparams,
            nonpadding=features_to_nonpadding(features, "inputs"))
        encoder_output = tf.expand_dims(encoder_output, 2)

        return encoder_output


@registry.register_model
class SimplifiedTransformerRegressor(SimplifiedTransformerEncoder):
    """Transformer inheriting from Encoder, for the regression problem.

    Final result is a tensor that has a shape of (?, 1, 1, 1).
    """

    def top(self, body_output, features):
        """Computes single scalar value from body_output."""

        with tf.variable_scope("reg_top_ffn"):
            x = body_output
            x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
            res = tf.layers.dense(x, 1, name="model_top")
            return res


def features_to_nonpadding(features, inputs_or_targets="inputs"):
    key = inputs_or_targets + "_segmentation"
    if features and key in features:
        return tf.minimum(tf.to_float(features[key]), 1.0)
    return None


def transformer_prepare_decoder(targets, hparams, features=None):
    """Prepare one shard of the model for the decoder.

    Args:
      targets: a Tensor.
      hparams: run hyperparameters
      features: optionally pass the entire features dictionary as well.
        This is needed now for "packed" datasets.

    Returns:
      decoder_input: a Tensor, bottom of decoder stack
      decoder_self_attention_bias: a bias tensor for use in decoder self-attention
    """
    if hparams.causal_decoder_self_attention:
        # Causal attention.
        if hparams.prepend_mode == "prepend_inputs_full_attention":
            decoder_self_attention_bias = (
                common_attention.attention_bias_prepend_inputs_full_attention(
                    common_attention.embedding_to_padding(targets)))
        else:
            decoder_self_attention_bias = (
                common_attention.attention_bias_lower_triangle(
                    common_layers.shape_list(targets)[1]))
    else:
        # Full attention.
        decoder_padding = common_attention.embedding_to_padding(targets)
        decoder_self_attention_bias = (
            common_attention.attention_bias_ignore_padding(decoder_padding))

    if features and "targets_segmentation" in features:
        # "Packed" dataset - keep the examples from seeing each other.
        targets_segmentation = features["targets_segmentation"]
        targets_position = features["targets_position"]
        decoder_self_attention_bias += common_attention.attention_bias_same_segment(
            targets_segmentation, targets_segmentation)
    else:
        targets_position = None
    if hparams.proximity_bias:
        decoder_self_attention_bias += common_attention.attention_bias_proximal(
            common_layers.shape_list(targets)[1])
    decoder_input = common_layers.shift_right_3d(targets)
    if hparams.pos == "timing":
        if targets_position is not None:
            decoder_input = common_attention.add_timing_signal_1d_given_position(
                decoder_input, targets_position)
        else:
            decoder_input = common_attention.add_timing_signal_1d(decoder_input)
    elif hparams.pos == "emb":
        decoder_input = common_attention.add_positional_embedding(
            decoder_input, hparams.max_length, "targets_positional_embedding",
            targets_position)

    if hparams.activation_dtype == "bfloat16":
        decoder_self_attention_bias = tf.cast(decoder_self_attention_bias,
                                              tf.bfloat16)
    return (decoder_input, decoder_self_attention_bias)


def transformer_decoder(decoder_input,
                        encoder_output,
                        decoder_self_attention_bias,
                        encoder_decoder_attention_bias,
                        hparams,
                        cache=None,
                        decode_loop_step=None,
                        name="decoder",
                        nonpadding=None,
                        save_weights_to=None,
                        make_image_summary=True,
                        losses=None):
    """A stack of transformer layers.

    Args:
      decoder_input: a Tensor
      encoder_output: a Tensor
      decoder_self_attention_bias: bias Tensor for self-attention
        (see common_attention.attention_bias())
      encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
        (see common_attention.attention_bias())
      hparams: hyperparameters for model
      cache: dict, containing tensors which are the results of previous
          attentions, used for fast decoding.
      decode_loop_step: An integer, step number of the decoding loop.
          Only used for inference on TPU.
      name: a string
      nonpadding: optional Tensor with shape [batch_size, encoder_length]
        indicating what positions are not padding.  This is used
        to mask out padding in convolutional layers.  We generally only
        need this mask for "packed" datasets, because for ordinary datasets,
        no padding is ever followed by nonpadding.
      save_weights_to: an optional dictionary to capture attention weights
        for visualization; the weights tensor will be appended there under
        a string key created from the variable scope (including name).
      make_image_summary: Whether to make an attention image summary.
      losses: optional list onto which to append extra training losses

    Returns:
      y: a Tensors
    """
    x = decoder_input
    attention_dropout_broadcast_dims = (
        common_layers.comma_separated_string_to_integer_list(
            getattr(hparams, "attention_dropout_broadcast_dims", "")))

    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_NUM_HIDDEN_LAYERS,
        value=hparams.num_decoder_layers or hparams.num_hidden_layers,
        hparams=hparams)
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_ATTENTION_DROPOUT,
        value=hparams.attention_dropout,
        hparams=hparams)
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_ATTENTION_DENSE,
        value={
            "use_bias": "false",
            "num_heads": hparams.num_heads,
            "hidden_size": hparams.hidden_size
        },
        hparams=hparams)

    with tf.variable_scope(name):
        for layer in range(hparams.num_decoder_layers or hparams.num_hidden_layers):
            layer_name = "layer_%d" % layer
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.variable_scope(layer_name):
                with tf.variable_scope("self_attention"):
                    if hparams.simple_head:
                        y = simplehead_attention(
                            common_layers.layer_preprocess(x, hparams),
                            None,
                            decoder_self_attention_bias,
                            hparams.attention_key_channels or hparams.hidden_size,
                            hparams.attention_value_channels or hparams.hidden_size,
                            hparams.hidden_size,
                            hparams.num_heads,
                            hparams.attention_dropout,
                            attention_type=hparams.self_attention_type,
                            max_relative_position=hparams.max_relative_position,
                            heads_share_relative_embedding=(
                              hparams.heads_share_relative_embedding),
                            add_relative_to_values=hparams.add_relative_to_values,
                            save_weights_to=save_weights_to,
                            cache=layer_cache,
                            make_image_summary=make_image_summary,
                            dropout_broadcast_dims=attention_dropout_broadcast_dims,
                            max_length=hparams.get("max_length"),
                            decode_loop_step=decode_loop_step,
                            vars_3d=hparams.get("attention_variables_3d"))
                    else:
                        y = common_attention.multihead_attention(
                            common_layers.layer_preprocess(x, hparams),
                            None,
                            decoder_self_attention_bias,
                            hparams.attention_key_channels or hparams.hidden_size,
                            hparams.attention_value_channels or hparams.hidden_size,
                            hparams.hidden_size,
                            hparams.num_heads,
                            hparams.attention_dropout,
                            attention_type=hparams.self_attention_type,
                            max_relative_position=hparams.max_relative_position,
                            heads_share_relative_embedding=(
                                hparams.heads_share_relative_embedding),
                            add_relative_to_values=hparams.add_relative_to_values,
                            save_weights_to=save_weights_to,
                            cache=layer_cache,
                            make_image_summary=make_image_summary,
                            dropout_broadcast_dims=attention_dropout_broadcast_dims,
                            max_length=hparams.get("max_length"),
                            decode_loop_step=decode_loop_step,
                            vars_3d=hparams.get("attention_variables_3d"))

                    x = common_layers.layer_postprocess(x, y, hparams)
                if encoder_output is not None:
                    with tf.variable_scope("encdec_attention"):
                        y = common_attention.multihead_attention(
                            common_layers.layer_preprocess(x, hparams),
                            encoder_output,
                            encoder_decoder_attention_bias,
                            hparams.attention_key_channels or hparams.hidden_size,
                            hparams.attention_value_channels or hparams.hidden_size,
                            hparams.hidden_size,
                            hparams.num_heads,
                            hparams.attention_dropout,
                            max_relative_position=hparams.max_relative_position,
                            heads_share_relative_embedding=(
                                hparams.heads_share_relative_embedding),
                            add_relative_to_values=hparams.add_relative_to_values,
                            save_weights_to=save_weights_to,
                            cache=layer_cache,
                            make_image_summary=make_image_summary,
                            dropout_broadcast_dims=attention_dropout_broadcast_dims,
                            max_length=hparams.get("max_length"),
                            vars_3d=hparams.get("attention_variables_3d"))
                        x = common_layers.layer_postprocess(x, y, hparams)
                if hparams.ffn:
                    with tf.variable_scope("ffn"):
                        y = transformer_ffn_layer(
                            common_layers.layer_preprocess(x, hparams),
                            hparams,
                            conv_padding="LEFT",
                            nonpadding_mask=nonpadding,
                            losses=losses,
                            cache=layer_cache,
                            decode_loop_step=decode_loop_step)
                        x = common_layers.layer_postprocess(x, y, hparams)
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        mlperf_log.transformer_print(
            key=mlperf_log.MODEL_HP_NORM,
            value={"hidden_size": hparams.hidden_size},
            hparams=hparams)
        return common_layers.layer_preprocess(x, hparams)


@registry.register_hparams
def simplified_transformer_base():
    hparams = transformer_base()
    hparams.simple_head = True
    hparams.ffn = False
    return hparams
