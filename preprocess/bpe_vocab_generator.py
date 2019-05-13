"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensor2tensor.data_generators import generator_utils
from SpmTextEncoder import SpmTextEncoder

import tensorflow as tf
import argparse
import shutil

_OD_TRAIN_DATASETS = [[
    "train.tgz", [
        "train/ja_zh.ja.tok.total.train",
        "train/ja_zh.zh.tok.total.train"
    ]
]]

#_ID_TRAIN_DATASETS = [[
#    "fine_tune_train.tgz", [
#        "fine_tune_train/en.tok.total.train",
#        "fine_tune_train/zh.tok.total.train"
#    ]
#]]

# Test set from News Commentary. 2000 lines
_OD_TEST_DATASETS = [[
    "test.tgz",
    ["test/ja_zh.ja.tok.total.test",
     "test/ja_zh.zh.tok.total.test"]
]]

#_ID_TEST_DATASETS = [[
#    "fine_tune_test.tgz", [
#        "fine_tune_test/en.tok.total.test",
#        "fine_tune_test/zh.tok.total.test"
#    ]
#]]


def get_filename(dataset):
    return dataset[0][0].split("/")[-1]


def get_dataset(tmp_dir):
    full_dataset = _OD_TRAIN_DATASETS
    for dataset in [_OD_TEST_DATASETS]:
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

    def generate_vocab(self, data_dir, tmp_dir, **kwargs):
        return


class T2TBpeVocabGenerator(BpeVocabGenerator):

    def __init__(self, name):
        super(T2TBpeVocabGenerator, self).__init__(name)

    @property
    def approx_vocab_size(self):
        return 50000

    @property
    def source_vocab_name(self):
        return "%s.en" % self.vocab_filename

    @property
    def target_vocab_name(self):
        return "%s.zh" % self.vocab_filename

    def generate_vocab(self, data_dir, tmp_dir, **kwargs):
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


class SpmBpeVocabGenerator(BpeVocabGenerator):

    def __init__(self, name):
        super(SpmBpeVocabGenerator, self).__init__(name)

    @property
    def approx_vocab_size(self):
        return 50000

    @property
    def source_vocab_name(self):
        return "%s.en" % self.vocab_filename

    @property
    def target_vocab_name(self):
        return "%s.zh" % self.vocab_filename

    def generate_vocab(self, data_dir, tmp_dir, **kwargs):
        datasets = get_dataset(tmp_dir)
        source_datasets = [[item[0], [item[1][0]]] for item in datasets]
        target_datasets = [[item[0], [item[1][1]]] for item in datasets]
        tf.gfile.MkDir(data_dir)
        for each in source_datasets:
            print("src_file:{file:s}".format(file=str(each)))

        for each in target_datasets:
            print("target_file:{file:s}".format(file=str(each)))

        source_vocab_generator = \
            generator_utils.generate_lines_for_vocab(tmp_dir, source_datasets, file_byte_budget=1e10)
        target_vocab_generator = \
            generator_utils.generate_lines_for_vocab(tmp_dir, target_datasets, file_byte_budget=1e10)

        if tf.gfile.Exists(os.path.join(tmp_dir,
                                        "{prefix:s}.corpus.txt".format(prefix=self.source_vocab_name))):
            print("====Source File Exists====")
        else:
            count = 0
            with tf.gfile.Open(os.path.join(tmp_dir,
                                            "{prefix:s}.corpus.txt".format(prefix=self.source_vocab_name)), "w") as f:
                for line in source_vocab_generator:
                    f.write(line)
                    f.write("\n")
                    count += 1
                print("====src:{src:d}=====".format(src=count))

        if tf.gfile.Exists(os.path.join(tmp_dir,
                                        "{prefix:s}.corpus.txt".format(prefix=self.target_vocab_name))):
            print("====Source File Exists====")
        else:
            count = 0
            with tf.gfile.Open(os.path.join(tmp_dir,
                                            "{prefix:s}.corpus.txt".format(prefix=self.target_vocab_name)), "w") as f:
                for line in target_vocab_generator:
                    f.write(line)
                    f.write("\n")
                    count += 1
                print("====target:{target:d}=====".format(target=count))

        _ = SpmTextEncoder.build_from_file(output_dir=data_dir,
                                           filename=os.path.join(tmp_dir,
                                                                 "{prefix:s}.corpus.txt".
                                                                 format(prefix=self.source_vocab_name)),
                                           vocab_size=self.approx_vocab_size,
                                           model_prefix=self.source_vocab_name,
                                           reserved_tokens=kwargs['reserved_tokens'],
                                           model_type=kwargs["model_type"])

        _ = SpmTextEncoder.build_from_file(output_dir=data_dir,
                                           filename=os.path.join(tmp_dir,
                                                                 "{prefix:s}.corpus.txt".
                                                                 format(prefix=self.target_vocab_name)),
                                           vocab_size=int(self.approx_vocab_size/2),
                                           model_prefix=self.target_vocab_name,
                                           reserved_tokens=kwargs['reserved_tokens'],
                                           model_type=kwargs["model_type"])


class SpmSameBpeVocabGenerator(BpeVocabGenerator):

    def __init__(self, name):
        super(SpmSameBpeVocabGenerator, self).__init__(name)

    @property
    def approx_vocab_size(self):
        return 32000

    @property
    def source_vocab_name(self):
        return "%s.ja" % self.vocab_filename

    @property
    def target_vocab_name(self):
        return "%s.zh" % self.vocab_filename

    def generate_vocab(self, data_dir, tmp_dir, **kwargs):
        datasets = get_dataset(tmp_dir)
        source_datasets = [[item[0], [item[1][0]]] for item in datasets]
        target_datasets = [[item[0], [item[1][1]]] for item in datasets]
        tf.gfile.MkDir(data_dir)
        for each in source_datasets:
            print("src_file:{file:s}".format(file=str(each)))

        for each in target_datasets:
            print("target_file:{file:s}".format(file=str(each)))

        source_vocab_generator = \
            generator_utils.generate_lines_for_vocab(tmp_dir, source_datasets, file_byte_budget=1e10)
        target_vocab_generator = \
            generator_utils.generate_lines_for_vocab(tmp_dir, target_datasets, file_byte_budget=1e10)

        if tf.gfile.Exists(os.path.join(tmp_dir,
                                        "{prefix:s}.corpus.txt".format(prefix=self.source_vocab_name))):
            print("====Source/Target File Exists====")
        else:
            count = 0
            with tf.gfile.Open(os.path.join(tmp_dir,
                                            "{prefix:s}.corpus.txt".format(prefix=self.source_vocab_name)), "w") as f:
                for src, tgt in zip(source_vocab_generator, target_vocab_generator):
                    f.write(src)
                    f.write("\n")
                    f.write(tgt)
                    f.write("\n")
                    count += 1
                print("====src/tgt:{src:d}=====".format(src=count))

        _ = SpmTextEncoder.build_from_file(output_dir=data_dir,
                                           filename=os.path.join(tmp_dir,
                                                                 "{prefix:s}.corpus.txt".
                                                                 format(prefix=self.source_vocab_name)),
                                           vocab_size=self.approx_vocab_size,
                                           model_prefix=self.source_vocab_name,
                                           reserved_tokens=kwargs['reserved_tokens'],
                                           model_type=kwargs["model_type"],
                                           sentence_size=40000000)

        model_src_path = os.path.join(data_dir, self.source_vocab_name + '.model')
        vocab_src_path = os.path.join(data_dir, self.source_vocab_name + '.vocab')

        model_tgt_path = os.path.join(data_dir, self.target_vocab_name + '.model')
        vocab_tgt_path = os.path.join(data_dir, self.target_vocab_name + '.vocab')

        shutil.copy(model_src_path, model_tgt_path)
        shutil.copy(vocab_src_path, vocab_tgt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bpe')
    parser.add_argument('--method', type=str, default="spm",
                        choices=["t2t", "spm", "sspm"],
                        help='method for bpe')
    parser.add_argument('--tmp-dir', type=str, required=True,
                        help='where the original data is')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='where to store model and vocab')
    parser.add_argument('--model-type', type=str, default="bpe",
                        choices=["bpe", "unigram", "char", "word"],
                        help='model_type for spm')
    parser.add_argument('--problem', type=str, default="translate_enzh_ai50k",
                        help="name for problem")
    parser.add_argument('--reserved-tokens', type=str, default=None,
                        help="reserved tokens")
    args = parser.parse_args()

    if args.method == "spm":
        generator = SpmBpeVocabGenerator(args.problem)
        generator.generate_vocab(args.data_dir,
                                 args.tmp_dir,
                                 model_type=args.model_type,
                                 reserved_tokens=args.reserved_tokens)
    elif args.method == "t2t":
        generator = T2TBpeVocabGenerator(args.problem)
        generator.generate_vocab(args.data_dir, args.tmp_dir)

    else:
        generator = SpmSameBpeVocabGenerator(args.problem)
        generator.generate_vocab(args.data_dir,
                                 args.tmp_dir,
                                 model_type=args.model_type,
                                 reserved_tokens=args.reserved_tokens)
