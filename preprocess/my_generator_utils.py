import tensorflow as tf
import os
from my_text_encoder import SubwordTextEncoder
from tensor2tensor.data_generators.generator_utils import generate_lines_for_vocab


def get_or_generate_vocab(data_dir, tmp_dir, vocab_filename, vocab_size,
                          sources, file_byte_budget=1e6,
                          max_subtoken_length=None, reserved_tokens=None):
    """Generate a vocabulary from the datasets in sources."""

    vocab_generator = generate_lines_for_vocab(tmp_dir, sources, file_byte_budget)
    return get_or_generate_vocab_inner(data_dir, vocab_filename, vocab_size,
                                       vocab_generator, max_subtoken_length, reserved_tokens)


def get_or_generate_vocab_inner(data_dir, vocab_filename, vocab_size,
                                generator, max_subtoken_length=None,
                                reserved_tokens=None):
    """Inner implementation for vocab generators.

    Args:
      data_dir: The base directory where data and vocab files are stored. If None,
        then do not save the vocab even if it doesn't exist.
      vocab_filename: relative filename where vocab file is stored
      vocab_size: target size of the vocabulary constructed by SubwordTextEncoder
      generator: a generator that produces tokens from the vocabulary
      max_subtoken_length: an optional integer.  Set this to a finite value to
        avoid quadratic costs during vocab building.
      reserved_tokens: List of reserved tokens. `text_encoder.RESERVED_TOKENS`
        should be a prefix of `reserved_tokens`. If `None`, defaults to
        `RESERVED_TOKENS`.

    Returns:
      A SubwordTextEncoder vocabulary object.
    """
    if data_dir and vocab_filename:
        vocab_filepath = os.path.join(data_dir, vocab_filename)
        if tf.gfile.Exists(vocab_filepath):
            tf.logging.info("Found vocab file: %s", vocab_filepath)
            return SubwordTextEncoder(vocab_filepath)
    else:
        vocab_filepath = None

    tf.logging.info("Generating vocab file: %s", vocab_filepath)
    vocab = SubwordTextEncoder.build_from_generator(
        generator, vocab_size, max_subtoken_length=max_subtoken_length,
        reserved_tokens=reserved_tokens)

    if vocab_filepath:
        tf.gfile.MakeDirs(data_dir)
        vocab.store_to_file(vocab_filepath)

    return vocab
