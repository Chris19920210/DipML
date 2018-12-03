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
from mosestokenizer import MosesTokenizer
import re
import simplejson as json
import logging
import time
import html


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
        self.delimiter = re.compile("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s")
        self.en_pattern = re.compile("^[a-zA-Z]+$")

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
            batch_output = serving_utils.predict(inputs[i:(i+FLAGS.batch)],
                                                 self.problem, self.request_fn)
            batch_output = [self.simple_formatter(output[0]) for output in batch_output]
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

    def english_token_eval(self, token):
        if re.match(self.en_pattern, token) is not None:
            return True

    def simple_formatter(self, sentence):
        sentence = sentence.split(" ")
        ret = ""
        for i in range(len(sentence)):
            if i < len(sentence) - 1:
                if self.english_token_eval(sentence[i]) and self.english_token_eval(sentence[i+1]):
                    ret += sentence[i]
                    ret += " "
                else:
                    ret += sentence[i]
            else:
                ret += sentence[i]
        return ret


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route("/translation", methods=['POST', 'GET'])
def translation():
    global nmt_client
    try:
        data = json.loads(str(request.get_data(), "utf-8").replace('\r\n', '').replace('\n', ''))["data"]
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
