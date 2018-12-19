import os

import nmt_serving_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import usr_dir
import tensorflow as tf
from mosestokenizer import MosesTokenizer
from mosestokenizer import MosesDetokenizer
import logging
import time
import html
import jieba


def validate_flags(server, servable_name):
    """Validates flags are set to acceptable values."""
    assert server
    assert servable_name


def make_request_fn(server, servable_name, timeout_secs):
    request_fn = nmt_serving_utils.make_grpc_request_fn(
        servable_name=servable_name,
        server=server,
        timeout_secs=timeout_secs)
    return request_fn


class EnZhNmtClient(object):
    def __init__(self,
                 server,
                 servable_name,
                 t2t_usr_dir,
                 problem,
                 data_dir,
                 timeout_secs):
        tf.logging.set_verbosity(tf.logging.INFO)
        validate_flags(server, servable_name)
        usr_dir.import_usr_dir(t2t_usr_dir)
        self.problem = registry.problem(problem)
        self.hparams = tf.contrib.training.HParams(
            data_dir=os.path.expanduser(data_dir))
        self.problem.get_hparams(self.hparams)
        self.request_fn = make_request_fn(server, servable_name, timeout_secs)
        self.tokenizer = MosesTokenizer('en')
        self.detokenizer = MosesDetokenizer('ko')
        fname = "inputs" if self.problem.has_inputs else "targets"
        self.input_encoder = self.problem.feature_info[fname].encoder
        self.output_decoder = self.problem.feature_info["targets"].encoder

    def sentence_prepare(self, sentence):
        sentence = self.tokenizer(sentence.strip())
        return len(sentence), html.unescape(" ".join(sentence).replace("@-@", "-"))

    def query(self, msg):
        """
        :param msg: dictionary{doc_id:string, batch_num:int, data:[{}]}
        :return:
        """
        doc_id = msg["document_id"]
        batch_num = msg["batch_num"]
        sentences = msg["data"]
        tmp = list(map(lambda x: (x["key"], self.sentence_prepare(x["value"])), sentences))
        tokens = sum(map(lambda x: x[1][0], tmp))
        sentences = list(map(lambda x: x[1][1], tmp))
        keys = list(map(lambda x: x[0], tmp))
        del tmp
        start = time.time()
        outputs = nmt_serving_utils.predict(sentences,
                                            self.problem,
                                            self.request_fn,
                                            self.input_encoder,
                                            self.output_decoder)
        outputs = [{"key": key, "value": self.detokenizer(zh[0].split(" "))} for key, zh in zip(keys, outputs)]
        end = time.time()
        printstr = "Sentences: {sentence:d}" \
                   "\tTokens: {tokens:d}" \
                   "\tTime: {time:.3f}ms" \
                   "\tToken/time: {per:.3f}ms"

        logging.info(printstr.format(sentence=len(sentences),
                                     tokens=tokens,
                                     time=(end - start) * 1000,
                                     per=((end - start) * 1000 / tokens)))

        return {"document_id": doc_id,
                "batch_num": batch_num,
                "data": outputs}


class ZhEnNmtClient(object):
    def __init__(self,
                 server,
                 servable_name,
                 t2t_usr_dir,
                 problem,
                 data_dir,
                 user_dict,
                 timeout_secs):
        tf.logging.set_verbosity(tf.logging.INFO)
        validate_flags(server, servable_name)
        usr_dir.import_usr_dir(t2t_usr_dir)
        self.problem = registry.problem(problem)
        self.hparams = tf.contrib.training.HParams(
            data_dir=os.path.expanduser(data_dir))
        self.problem.get_hparams(self.hparams)
        self.request_fn = make_request_fn(server, servable_name, timeout_secs)
        jieba.load_userdict(user_dict)
        self.detokenizer = MosesDetokenizer('en')
        fname = "inputs" if self.problem.has_inputs else "targets"
        self.input_encoder = self.problem.feature_info[fname].encoder
        self.output_decoder = self.problem.feature_info["targets"].encoder

    def sentence_prepare(self, sentence):
        sentence = jieba.lcut(sentence.strip())
        return len(sentence), " ".join(sentence)

    def query(self, msg):
        """
        :param msg: dictionary{doc_id:string, batch_num:int, data:[{}]}
        :return:
        """
        doc_id = msg["document_id"]
        batch_num = msg["batch_num"]
        sentences = msg["data"]
        tmp = list(map(lambda x: (x["key"], self.sentence_prepare(x["value"])), sentences))
        tokens = sum(map(lambda x: x[1][0], tmp))
        sentences = list(map(lambda x: x[1][1], tmp))
        keys = list(map(lambda x: x[0], tmp))
        del tmp
        start = time.time()
        outputs = nmt_serving_utils.predict(sentences,
                                            self.problem,
                                            self.request_fn,
                                            self.input_encoder,
                                            self.output_decoder)
        outputs = [{"key": key, "value": self.detokenizer(en[0].split(" "))} for key, en in zip(keys, outputs)]
        end = time.time()
        printstr = "Sentences: {sentence:d}" \
                   "\tTokens: {tokens:d}" \
                   "\tTime: {time:.3f}ms" \
                   "\tToken/time: {per:.3f}ms"

        logging.info(printstr.format(sentence=len(sentences),
                                     tokens=tokens,
                                     time=(end - start) * 1000,
                                     per=((end - start) * 1000 / tokens)))

        return {"document_id": doc_id,
                "batch_num": batch_num,
                "data": outputs}

