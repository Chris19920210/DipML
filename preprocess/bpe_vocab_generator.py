"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensor2tensor.data_generators import generator_utils

import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='BPE generator')
parser.add_argument('--tmp-dir', type=str, default=None,
                    help='path to tmp-dir(where the data is)')
parser.add_argument('--data-dir', type=str, default=None,
                    help='path to data-dir(where to store the vocab)')
args = parser.parse_args()


_OD_TRAIN_DATASETS = [[
    "train.tgz", [
        "train/ai_challenger_en-zh.en.tok.train",
        "train/ai_challenger_en-zh.zh.tok.train"
    ]
]]

_ID_TRAIN_DATASETS = [[
    "fine_tune_train.tgz", [
        "fine_tune_train/en.tok.train",
        "fine_tune_train/zh.tok.train"
    ]
]]

# Test set from News Commentary. 2000 lines
_OD_TEST_DATASETS = [[
    "test.tgz",
    ["test/ai_challenger_en-zh.en.tok.test",
     "test/ai_challenger_en-zh.zh.tok.test"]
]]

_ID_TEST_DATASETS = [[
    "fine_tune_test.tgz", [
        "fine_tune_test/en.tok.test",
        "fine_tune_test/zh.tok.test"
    ]
]]


def get_filename(dataset):
    return dataset[0][0].split("/")[-1]


def get_dataset(tmp_dir):
    full_dataset = _OD_TRAIN_DATASETS
    for dataset in [_OD_TEST_DATASETS, _ID_TRAIN_DATASETS, _ID_TEST_DATASETS]:
        filename = get_filename(dataset)
        tmp_filepath = os.path.join(tmp_dir, filename)
        if tf.gfile.Exists(tmp_filepath):
            full_dataset += dataset
        else:
            tf.logging.info("[TranslateEzhWmt] dataset incomplete, "
                            "you need to manually download %s" % filename)
    return full_dataset


class BpeVocabGenerator(object):
    def __init__(self, name):
        self.name = name

    @property
    def approx_vocab_size(self):
        return 50000

    @property
    def vocab_filename(self):
        return "vocab.%s.%d.%s" % (self.name,
                                   self.approx_vocab_size,
                                   "subwords")

    def generate_vocab(self, data_dir, tmp_dir):
        return


class T2TBpeVocabGenerator(BpeVocabGenerator):

    def __init__(self, name):
        super(T2TBpeVocabGenerator).__init__(name)

    @property
    def approx_vocab_size(self):
        return 50000

    @property
    def source_vocab_name(self):
        return "%s.en" % self.vocab_filename

    @property
    def target_vocab_name(self):
        return "%s.zh" % self.vocab_filename

    def generate_vocab(self, data_dir, tmp_dir):
        datasets = get_dataset(tmp_dir)
        source_datasets = [[item[0], [item[1][0]]] for item in datasets]
        target_datasets = [[item[0], [item[1][1]]] for item in datasets]
        _ = generator_utils.get_or_generate_vocab(
            data_dir,
            tmp_dir,
            self.source_vocab_name,
            self.approx_vocab_size,
            source_datasets,
            file_byte_budget=1e8)
        _ = generator_utils.get_or_generate_vocab(
            data_dir,
            tmp_dir,
            self.target_vocab_name,
            int(self.approx_vocab_size / 2),
            target_datasets,
            file_byte_budget=1e8)


if __name__ == '__main__':
    generator = T2TBpeVocabGenerator('translate_enzh_ai50k')
    generator.generate_vocab(args.data_dir, args.tmp_dir)
