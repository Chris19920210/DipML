from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators.text_encoder import TextEncoder, strip_ids
import os
import sentencepiece as spm
from sentencepiece import SentencePieceProcessor
import shutil
import numpy as np

EOS = "</s>"
UNK = "<unk>"
PAD = "<pad>"
RESERVED_TOKENS = [PAD, EOS, UNK]
NUM_RESERVED_TOKENS = len(RESERVED_TOKENS)
UNK_ID = RESERVED_TOKENS.index(UNK)
PAD_ID = RESERVED_TOKENS.index(PAD)
EOS_ID = RESERVED_TOKENS.index(EOS)


class PickalableSWIG:

    def __setstate__(self, state):
        self.__init__(*state['args'])

    def __getstate__(self):
        return {'args': self.args}


class MySentencePieceProcessor(SentencePieceProcessor, PickalableSWIG):
    def __init__(self, *args):
        self.args = args
        SentencePieceProcessor.__init__(self)


class SpmTextEncoder(TextEncoder):

    def __init__(self, filename=None):
        """Initialize and read from a file, if provided.

        Args:
          filename: filename from which to load the model. If None, do not load a
            model
        """
        self.filename = filename
        self.sp = MySentencePieceProcessor()
        if filename is not None:
            self.sp.Load(filename)
        super(SpmTextEncoder, self).__init__(num_reserved_ids=NUM_RESERVED_TOKENS)

    def encode(self, s):
        """Converts a native string to a list of subtoken ids.
        Args:
        s: a native string.
        Returns:
            a list of integers in the range [0, vocab_size)
        """
        return self.sp.encode_as_ids(s)

    def decode(self, ids, strip_extraneous=False):
        """Converts a sequence of subtoken ids to a native string.
        Args:
            ids: a list of integers in the range [0, vocab_size)
            strip_extraneous: bool, whether to strip off extraneous tokens
            (EOS and PAD).
        Returns:
            a native string
        """
        if strip_extraneous:
            ids = strip_ids(ids, list(range(self._num_reserved_ids or 0)))
        if isinstance(ids, list):
            return self.sp.DecodeIds(list(map(lambda x: int(x), ids)))
        return self.sp.DecodeIds(ids.tolist())

    def decode_list(self, ids):
        return [self._subtoken_id_to_subtoken_string(s) for s in ids]

    @property
    def vocab_size(self):
        """The subtoken vocabulary size."""
        return self.sp.GetPieceSize()

    def _subtoken_id_to_subtoken_string(self, subtoken):
        """Converts a subtoken integer ID to a subtoken string."""
        if 0 <= subtoken < self.vocab_size:
            return self.sp.IdToPiece(int(subtoken))
        return u""

    def encode_as_pieces(self, s):
        """ encode sentence as sentence piece
        :param s: string
        :return: list[str] pieces list
        """
        return self.sp.EncodeAsPieces(s)

    @classmethod
    def build_from_file(cls,
                        output_dir,
                        filename,
                        vocab_size,
                        model_prefix="m",
                        reserved_tokens=None,
                        model_type="unigram",
                        sentence_size=20000000):
        """
        :param filename: file input to train the model
        :param output_dir: where to output the model
        :param vocab_size: vocab size
        :param reserved_tokens:list of tokens to be reserved if None the default reserved
        :param model_type: unigram/bpe/char/word
        :param sentence_size: sentence size for training
        :param model_prefix: model_prefix
        :return:
        """
        if reserved_tokens is None:
            args = "--bos_id=-1 " \
                   "--pad_id=0 " \
                   "--eos_id=1 " \
                   "--unk_id=2 " \
                   "--input={filename:s} " \
                   "--model_prefix={model_prefix:s} " \
                   "--model_type={model_type:s} " \
                   "--vocab_size={vocab_size:d} " \
                   "--character_coverage=1.0 " \
                   "--input_sentence_size={sentence_size:d}"\
                .format(filename=filename,
                        model_type=model_type,
                        vocab_size=vocab_size,
                        sentence_size=sentence_size,
                        model_prefix=model_prefix)
        else:
            args = "--bos_id=-1 " \
                   "--pad_id=0 " \
                   "--eos_id=1 " \
                   "--unk_id=2 " \
                   "--user_defined_symbols={reserved_tokens:s} " \
                   "--input={filename:s} " \
                   "--model_prefix={model_prefix:s} " \
                   "--model_type={model_type:s} " \
                   "--vocab_size={vocab_size:d} " \
                   "--character_coverage=1.0 " \
                   "--input_sentence_size={sentence_size:d}"\
                .format(filename=filename,
                        reserved_tokens=",".join(reserved_tokens),
                        model_type=model_type,
                        vocab_size=vocab_size,
                        sentence_size=sentence_size,
                        model_prefix=model_prefix)
        spm.SentencePieceTrainer.Train(args)
        model_path = os.path.join(os.getcwd(), model_prefix+'.model')
        vocab_path = os.path.join(os.getcwd(), model_prefix+'.vocab')
        model_dest_path = os.path.join(output_dir, model_prefix+'.model')
        vocab_dest_path = os.path.join(output_dir, model_prefix+'.vocab')
        shutil.move(model_path, model_dest_path)
        shutil.move(vocab_path, vocab_dest_path)
        return cls(filename=model_dest_path)


if __name__ == '__main__':
    spm_encoder = SpmTextEncoder.build_from_file("/home/chris",
                                                 "/home/chris/nmt/zh.tok.total.text",
                                                 8000,
                                                 model_prefix="dd",
                                                 model_type="bpe")
    a = spm_encoder.decode(np.array([579, 4831, 1903, 4207, 1903, 2160, 3624, 1564, 4598]))
    print(a)
