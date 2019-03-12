from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, request, jsonify
from flask_cors import CORS

import os

from tensor2tensor.serving import serving_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import usr_dir
import tensorflow as tf
from mosestokenizer import MosesTokenizer, MosesDetokenizer
from zhon.hanzi import punctuation as zhpunc
import re
import simplejson as json
import logging
import time
import html
from tensor2tensor.data_generators import text_encoder
import functools


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("server", None, "Address to Tensorflow Serving server.")
flags.DEFINE_string("servable_name", None, "Name of served model.")
flags.DEFINE_string("problem", None, "Problem name.")
flags.DEFINE_string("data_dir", None, "Data directory, for vocab files.")
flags.DEFINE_string("t2t_usr_dir", None, "Usr dir for registrations.")
flags.DEFINE_integer("timeout_secs", 100, "Timeout for query.")
flags.DEFINE_integer("batch", 5, "Batch size for request")
flags.DEFINE_string("port", None, "Port")
flags.DEFINE_string("host", None, "host")

app = Flask(__name__)
CORS(app)


class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


def gnmt_make_example(input_ids, problem, input_feature_name="inputs"):
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


def validate_flags():
    """Validates flags are set to acceptable values."""
    assert FLAGS.server
    assert FLAGS.servable_name
    assert FLAGS.port


def make_request_fn():
    request_fn = serving_utils.make_grpc_request_fn(
        servable_name=FLAGS.servable_name,
        server=FLAGS.server,
        timeout_secs=FLAGS.timeout_secs)
    return request_fn


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


def predict(inputs_list, problem, request_fn):
    assert isinstance(inputs_list, list)
    fname = "sources"
    input_encoder = problem.feature_info["inputs"].encoder
    input_ids_list = [
        encode(inputs, input_encoder, add_eos=False)
        for inputs in inputs_list
    ]
    examples = [gnmt_make_example(input_ids, problem, fname)
                for input_ids in input_ids_list]
    predictions = request_fn(examples)
    output_decoder = problem.feature_info["targets"].encoder
    outputs = [(gnmt_decode(prediction["outputs"], output_decoder), prediction["scores"])
               for prediction in predictions]

    return outputs


class NmtClient(object):
    def __init__(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        validate_flags()
        usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
        self.problem = registry.problem(FLAGS.problem)
        self.hparams = tf.contrib.training.HParams(
            data_dir=os.path.expanduser(FLAGS.data_dir))
        self.problem.get_hparams(self.hparams)
        self.request_fn = make_request_fn()
        self.tokenizer = MosesTokenizer('en')
        self.moses_detokenizer = MosesDetokenizer('zh')
        self.delimiter = re.compile("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s")

    def detokenizer(self, sentence):
        return re.sub(r'\s+([{0}])\s+'.format(zhpunc), r"\1", self.moses_detokenizer(sentence))

    def query(self, sentences):
        """
        :param sentences: str
        :return:
        """
        inputs = re.split(self.delimiter, sentences.strip())
        tmp = []
        tokens = 0
        for sentence in inputs:
            sentence = self.tokenizer(sentence.strip())
            tokens += len(sentence)
            tmp.append(html.unescape(" ".join(sentence).replace("@-@", "-")))
        inputs = tmp
        del tmp
        outputs = []
        start = time.time()
        for i in range(0, len(inputs), FLAGS.batch):
            batch_output = predict(inputs[i:(i+FLAGS.batch)], self.problem, self.request_fn)
            batch_output = [self.detokenizer(output[0].split(" ")) for output in batch_output]
            outputs.extend([{"key": en, "value": zh}
                            for en, zh in zip(inputs[i:(i+FLAGS.batch)], batch_output)])
        end = time.time()
        printstr = "Sentences: {sentence:d}\tTokens: {tokens:d}" \
                   "\tTime: {time:.3f}ms\tTokens/time: {per:.3f}ms"
        logging.info(printstr.format(sentence=len(inputs),
                                     tokens=tokens,
                                     time=(end - start) * 1000,
                                     per=((end - start) * 1000 / tokens)))
        for output in outputs:
            logging.info("Input:{input:s}".format(input=output["key"]))
            logging.info("Output:{output:s}".format(output=output["value"]))
        return outputs


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route("/translation_gnmt", methods=['POST', 'GET'])
def translation():
    global nmt_client
    try:
        data = json.loads(request.get_data(), strict=False)["data"]
        print(request.get_data())
        return json.dumps({"data": nmt_client.query(data)}, indent=1, ensure_ascii=False)
    except Exception as e:
        logging.error(str(e))
        raise InvalidUsage('Ooops. Something went wrong', status_code=503)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename='./query.log',
                        filemode='w')
    flags.mark_flags_as_required(["problem", "data_dir"])
    nmt_client = NmtClient()
    print("Starting app...")
    app.run(host=FLAGS.host, threaded=True, port=FLAGS.port)
