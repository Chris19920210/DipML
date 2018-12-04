import tensorflow as tf
import os
from tensor2tensor.data_generators.generator_utils import generate_lines_for_vocab
from SpmTextEncoder import SpmTextEncoder


def get_or_generate_spm(data_dir,
                        tmp_dir,
                        vocab_size,
                        model_prefix,
                        sources,
                        file_byte_budget=1e6,
                        model_type="bpe",
                        reserved_tokens=None):
    """Generate a vocabulary from the datasets in sources."""
    vocab_generator = generate_lines_for_vocab(tmp_dir, sources, file_byte_budget)
    return get_or_generate_spm_inner(tmp_dir,
                                     data_dir,
                                     vocab_size,
                                     model_prefix,
                                     vocab_generator,
                                     model_type,
                                     reserved_tokens)


def get_or_generate_spm_inner(tmp_dir,
                              data_dir,
                              vocab_size,
                              model_prefix,
                              generator,
                              model_type,
                              reserved_tokens=None):
    """Inner implementation for vocab generators.

    Args:
      tmp_dir:where to store the data
      data_dir: The base directory where data and vocab files are stored. If None,
        then do not save the vocab even if it doesn't exist.
      model_type: unigram/bpe/char/word
      model_prefix:src/tar
      vocab_size: target size of the vocabulary constructed by SubwordTextEncoder
      generator: a generator that produces tokens from the vocabulary
      reserved_tokens: List of reserved tokens. `text_encoder.RESERVED_TOKENS`
        should be a prefix of `reserved_tokens`. If `None`, defaults to
        `RESERVED_TOKENS`.

    Returns:
      A SubwordTextEncoder vocabulary object.
    """
    if data_dir and model_prefix:
        model_path = os.path.join(data_dir, model_prefix+".model")
        if tf.gfile.Exists(model_path):
            tf.logging.info("Found model file: %s", model_path)
            return SpmTextEncoder(model_path)
    else:
        model_path = None

    tf.logging.info("Generating model file: %s", model_path)
    with tf.gfile.Open(os.path.join(tmp_dir,
                                    "{prefix:s}.corpus.txt".format(prefix=model_prefix)), "w") as f:
        for line in generator:
            f.write(line)
            f.write("\n")

    model = SpmTextEncoder.build_from_file(output_dir=data_dir,
                                           filename=os.path.join(tmp_dir,
                                                                 "{prefix:s}.corpus.txt"
                                                                 .format(prefix=model_prefix)),
                                           vocab_size=vocab_size,
                                           model_prefix=model_prefix,
                                           reserved_tokens=reserved_tokens,
                                           model_type=model_type)

    return model
