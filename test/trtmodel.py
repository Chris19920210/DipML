import tensorflow as tf
from tensor2tensor.utils import registry
from tensor2tensor.utils import usr_dir
import time
from test_utils import encode, decode, make_example
import os
from tensorflow.contrib import tensorrt as trt
from tensorflow.python.framework import ops

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("problem", None, "Problem name.")
flags.DEFINE_string("data_dir", None, "Data directory, for vocab files.")
flags.DEFINE_string("frozen_graph_filename", None, "model directory, for vocab files.")
flags.DEFINE_string("t2t_usr_dir", None, "usr_dir")
trt_gpu_ops = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)


def make_predict_fn(frozen_graph_filename):

    # Then, we import the graph_def into a new Graph and returns it
    ops.reset_default_graph()
    graph = tf.get_default_graph()
    with graph.as_default():
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        init_op = tf.group([tf.global_variables_initializer(), tf.local_variables_initializer(),
                            tf.tables_initializer()])
        tf.import_graph_def(
            graph_def, name="")
        trt_input = graph.get_tensor_by_name("serialized_example:0")
        trt_batch_prediction_key = graph.get_tensor_by_name("DatasetToSingleElement:0")
        trt_outputs = graph.get_tensor_by_name("transformer/strided_slice_10:0")
        trt_scores = graph.get_tensor_by_name("transformer/strided_slice_11:0")

    sess = tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=trt_gpu_ops))
    sess.run(init_op)

    def predict_fn(dataset):
        batch_prediction_key, outputs, scores = sess.run([trt_batch_prediction_key, trt_outputs, trt_scores],
                                                         feed_dict={
                                                             trt_input: dataset
                                                         })
        return {
            'batch_prediction_key': batch_prediction_key,
            'outputs': outputs,
            'scores': scores
        }

    return predict_fn


def predict(inputs_list, problem, predict_fn):
    """Encodes inputs, makes request to deployed TF model, and decodes outputs."""
    assert isinstance(inputs_list, list)
    fname = "inputs" if problem.has_inputs else "targets"
    input_encoder = problem.feature_info[fname].encoder
    encode_start = time.time()
    input_ids_list = [
        encode(inputs, input_encoder, add_eos=problem.has_inputs)
        for inputs in inputs_list
    ]
    encode_end = time.time()
    examples = [make_example(input_ids, problem, fname).SerializeToString()
                for input_ids in input_ids_list]
    predict_start = time.time()
    predictions = predict_fn(examples)
    predict_end = time.time()
    output_decoder = problem.feature_info["targets"].encoder
    decode_start = time.time()
    outputs = [(decode(output, output_decoder), score)
               for output, score in zip(predictions["outputs"], predictions["scores"])]
    decode_end = time.time()
    encode_time = (encode_end - encode_start) * 1000
    predict_time = (predict_end - predict_start) * 1000
    decode_time = (decode_end - decode_start) * 1000
    total_time = (decode_end - encode_start) * 1000
    print_str = """
  Batch:{batch:d} \t
  Encode:{encode:.3f} \t 
  Prediction:{predict:.3f} \t 
  Decode:{decode:.3f}　\t
  Total:{total:.3f}
  """
    print(print_str.format(batch=len(outputs),
                           encode=encode_time,
                           predict=predict_time,
                           decode=decode_time,
                           total=total_time))

    return outputs


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
    problem = registry.problem(FLAGS.problem)
    hparams = tf.contrib.training.HParams(
        data_dir=os.path.expanduser(FLAGS.data_dir))
    problem.get_hparams(hparams)
    predict_fn = make_predict_fn(FLAGS.frozen_graph_filename)
    while True:
        inputs = input(">> ")
        outputs = predict([inputs], problem, predict_fn)
        outputs, = outputs
        output, score = outputs
        print_str = """
Input:
{inputs}

Output (Score {score:.3f}):
{output}
    """
        print(print_str.format(inputs=inputs, output=output, score=score))


if __name__ == "__main__":
    flags.mark_flags_as_required(["problem", "data_dir"])
    tf.app.run()
