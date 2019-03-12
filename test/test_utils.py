import tensorflow as tf
import functools
from tensor2tensor.data_generators import text_encoder
import re


def make_example(input_ids, problem, input_feature_name="inputs"):
    """Make a tf.train.Example for the problem.

    features[input_feature_name] = input_ids

    Also fills in any other required features with dummy values.

    Args:
      input_ids: list<int>.
      problem: Problem.
      input_feature_name: name of feature for input_ids.

    Returns:
      tf.train.Example
    """
    features = {
        input_feature_name:
            tf.train.Feature(int64_list=tf.train.Int64List(value=input_ids))
    }

    # Fill in dummy values for any other required features that presumably
    # will not actually be used for prediction.
    data_fields, _ = problem.example_reading_spec()
    for fname, ftype in data_fields.items():
        if fname == input_feature_name:
            continue
        if not isinstance(ftype, tf.FixedLenFeature):
            # Only FixedLenFeatures are required
            continue
        if ftype.default_value is not None:
            # If there's a default value, no need to fill it in
            continue
        num_elements = functools.reduce(lambda acc, el: acc * el, ftype.shape, 1)
        if ftype.dtype in [tf.int32, tf.int64]:
            value = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[0] * num_elements))
        if ftype.dtype in [tf.float32, tf.float64]:
            value = tf.train.Feature(
                float_list=tf.train.FloatList(value=[0.] * num_elements))
        if ftype.dtype == tf.bytes:
            value = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[""] * num_elements))
        tf.logging.info("Adding dummy value for feature %s as it is required by "
                        "the Problem.", fname)
        features[fname] = value
    return tf.train.Example(features=tf.train.Features(feature=features))


def encode(inputs, encoder, add_eos=True):
    input_ids = encoder.encode(inputs)
    if add_eos:
        input_ids.append(text_encoder.EOS_ID)
    return input_ids


def decode(output_ids, output_decoder):
    return output_decoder.decode(output_ids, strip_extraneous=True)


def gnmt_decode(output_ids, output_decoder):
    output_ids = output_ids.tolist()

    tgt_eos = 2
    if tgt_eos in output_ids:
        output_ids = output_ids[:output_ids.index(tgt_eos)]

    return output_decoder.decode(output_ids, strip_extraneous=True)


delimiter = re.compile("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s")


def split_sentence(s):
    global delimiter
    s = re.split(delimiter, s.strip())
    return s



