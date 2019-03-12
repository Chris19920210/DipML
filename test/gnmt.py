from tensorflow.contrib import predictor
import tensorflow as tf
from tensor2tensor.utils import registry
from tensor2tensor.utils import usr_dir
import time
from test_utils import encode, gnmt_decode, make_example, split_sentence
import os
from mosestokenizer import MosesTokenizer
import html

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("problem", None, "Problem name.")
flags.DEFINE_string("data_dir", None, "Data directory, for vocab files.")
flags.DEFINE_string("export_dir", None, "model directory, for vocab files.")
flags.DEFINE_string("t2t_usr_dir", None, "usr_dir")


def make_predict_fn():
    predict_fn = predictor.from_saved_model(FLAGS.export_dir)
    return predict_fn


def predict(inputs_list, problem, predict_fn):
    """Encodes inputs, makes request to deployed TF model, and decodes outputs."""
    assert isinstance(inputs_list, list)
    fname = "sources"
    input_encoder = problem.feature_info["inputs"].encoder
    encode_start = time.time()
    input_ids_list = [
        encode(inputs, input_encoder, add_eos=False)
        for inputs in inputs_list
    ]
    encode_end = time.time()
    examples = [make_example(input_ids, problem, fname).SerializeToString()
                for input_ids in input_ids_list]
    examples = {'input': examples}
    predict_start = time.time()
    predictions = predict_fn(examples)
    predict_end = time.time()
    output_decoder = problem.feature_info["targets"].encoder
    decode_start = time.time()
    outputs = [(gnmt_decode(output, output_decoder), score)
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
    predict_fn = make_predict_fn()
    tokenizer = MosesTokenizer("en")
    while True:
        inputs = input(">> ")
        inputs = split_sentence(inputs)
        inputs = list(map(tokenizer, inputs))
        inputs = list(map(lambda input: html.unescape(" ".join(input).replace("@-@", "-")), inputs))
        outputs = predict(inputs, problem, predict_fn)
        outputs = list(map(lambda x: x[0], outputs))
        print_str = """
Input:
{inputs}

Output:
{output}
    """
        print(print_str.format(inputs="\n".join(inputs), output="\n".join(outputs)))


if __name__ == "__main__":
    flags.mark_flags_as_required(["problem", "data_dir"])
    tf.app.run()
