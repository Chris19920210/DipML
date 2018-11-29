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

"""Query an exported model. Py2 only. Install tensorflow-serving-api."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import input  # pylint: disable=redefined-builtin

from tensor2tensor.serving import serving_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import usr_dir
import tensorflow as tf
from mosestokenizer import MosesTokenizer
import re

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("server", None, "Address to Tensorflow Serving server.")
flags.DEFINE_string("servable_name", None, "Name of served model.")
flags.DEFINE_string("problem", None, "Problem name.")
flags.DEFINE_string("data_dir", None, "Data directory, for vocab files.")
flags.DEFINE_string("t2t_usr_dir", None, "Usr dir for registrations.")
flags.DEFINE_integer("timeout_secs", 10, "Timeout for query.")


def validate_flags():
    """Validates flags are set to acceptable values."""
    assert FLAGS.server
    assert FLAGS.servable_name


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

    def query(self, str):
        """
        :param input: str
        :return:
        """
        inputs = re.split(self.delimiter, str)
        inputs = [" ".join(self.tokenizer(sentence)) for sentence in inputs]
        outputs = serving_utils.predict(inputs, self.problem, self.request_fn)
        outputs = [output[0].replace(" ", "") for output in outputs]
        return "".join(outputs)


def main(_):
    nmt_client = NmtClient()
    while True:
        inputs = input(">> ")
        outputs = nmt_client.query(inputs)
        printstr = """
        Output:
        {output:s}
        """
        print(printstr.format(output=outputs))


if __name__ == "__main__":
  flags.mark_flags_as_required(["problem", "data_dir"])
  tf.app.run()
