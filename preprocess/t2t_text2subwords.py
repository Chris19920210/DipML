#!/usr/bin/env python3
from tensor2tensor.data_generators import text_encoder

import tensorflow as tf
import sys

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("vocab_src", None, "Path to the subword vocabulary")
flags.DEFINE_string("vocab_trg", None, "Path to the subword vocabulary")
flags.DEFINE_string("src", None, "Path to the source-language text")
flags.DEFINE_string("trg", None, "Path to the target-language text")
# TODO print the actual subwords, use vocab._subtoken_id_to_subtoken_string() instead of _subtoken_ids_to_tokens()
flags.DEFINE_bool("print", False, "Print a character for each subword?")


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def words_subwords(vocab, string):
    # subwords = vocab._subtoken_ids_to_tokens([x]) for x in vocab.encode(string)]
    n_words = len(string.split())
    n_subwords = len(vocab.encode(string))
    return n_words, n_subwords


s_words, t_words, m_words = 0, 0, 0
s_subws, t_subws, m_subws = 0, 0, 0
sents = 0


def print_stats():
    global s_words, t_words, m_words, s_subws, t_subws, m_subws, sents
    eprint("\ntotal: sents=%d words=%d subwords=%s subwords/words %.4f" % (sents, m_words, m_subws, m_subws / m_words))
    eprint("source: words=%d subwords=%d" % (s_words, s_subws))
    eprint("target: words=%d subwords=%d" % (t_words, t_subws))


def main(_):
    global s_words, t_words, m_words, s_subws, t_subws, m_subws, sents
    vocab_src = text_encoder.SubwordTextEncoder(FLAGS.vocab_src)
    vocab_trg = text_encoder.SubwordTextEncoder(FLAGS.vocab_trg)
    with open(FLAGS.src, encoding="utf-8") as src, open(FLAGS.trg, encoding="utf-8") as trg:
        for s, t in zip(src, trg):
            sents += 1
            s = s.strip()
            t = t.strip()
            s_w, s_s = words_subwords(vocab_src, s)
            t_w, t_s = words_subwords(vocab_trg, t)
            s_words += s_w
            t_words += t_w
            m_words += max(s_w, t_w)
            s_subws += s_s
            t_subws += t_s
            m_subws += max(s_s, t_s)
            if sents % 100000 == 0:
                print_stats()
            if FLAGS.print:
                print("a" * max(s_s, t_s))
    print_stats()


if __name__ == "__main__":
    tf.app.run()
from tensor2tensor.models.research import universal_transformer