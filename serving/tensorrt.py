from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
from tensorflow.contrib.saved_model.python.saved_model import reader
import tensorflow.contrib.tensorrt as trt

_GPU_MEM_FRACTION = 0.50
_GRAPH_FILE = "frozen_graph.pb"


def get_serving_meta_graph_def(savedmodel_dir):
    """Extract the SERVING MetaGraphDef from a SavedModel directory.
    Args:
      savedmodel_dir: the string path to the directory containing the .pb
        and variables for a SavedModel. This is equivalent to the subdirectory
        that is created under the directory specified by --export_dir when
        running an Official Model.
    Returns:
      MetaGraphDef that should be used for tag_constants.SERVING mode.
    Raises:
      ValueError: if a MetaGraphDef matching tag_constants.SERVING is not found.
    """
    # We only care about the serving graph def
    tag_set = set([tf.saved_model.tag_constants.SERVING])
    serving_graph_def = None
    saved_model = reader.read_saved_model(savedmodel_dir)
    for meta_graph_def in saved_model.meta_graphs:
        if set(meta_graph_def.meta_info_def.tags) == tag_set:
            serving_graph_def = meta_graph_def
    if not serving_graph_def:
        raise ValueError("No MetaGraphDef found for tag_constants.SERVING. "
                         "Please make sure the SavedModel includes a SERVING def.")

    return serving_graph_def


def write_graph_to_file(graph_name, graph_def, output_dir):
    """Write Frozen Graph file to disk."""
    output_path = os.path.join(output_dir, graph_name)
    with tf.gfile.GFile(output_path, "wb") as f:
        f.write(graph_def.SerializeToString())


def convert_savedmodel_to_frozen_graph(savedmodel_dir, output_dir):
    """Convert a SavedModel to a Frozen Graph.
    A SavedModel includes a `variables` directory with variable values,
    and a specification of the MetaGraph in a ProtoBuffer file. A Frozen Graph
    takes the variable values and inserts them into the graph, such that the
    SavedModel is all bundled into a single file. TensorRT and TFLite both
    leverage Frozen Graphs. Here, we provide a simple utility for converting
    a SavedModel into a frozen graph for use with these other tools.
    Args:
      savedmodel_dir: the string path to the directory containing the .pb
        and variables for a SavedModel. This is equivalent to the subdirectory
        that is created under the directory specified by --export_dir when
        running an Official Model.
      output_dir: string representing path to the output directory for saving
        the frozen graph.
    Returns:
      Frozen Graph definition for use.
    """
    meta_graph_def = get_serving_meta_graph_def(savedmodel_dir)
    signature_def = meta_graph_def.signature_def[
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    outputs = [v.name for v in signature_def.outputs.values()]
    output_names = [node.split(":")[0] for node in outputs]

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        tf.saved_model.loader.load(
            sess, meta_graph_def.meta_info_def.tags, savedmodel_dir)
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), output_names,
            variable_names_blacklist=
            ['transformer/symbol_modality_25000_512/target_emb/weights_{}'.format(z) for z in range(16)] +
            ["transformer/body/decoder/layer_{}/self_attention/layer_prepostprocess/layer_norm/layer_norm_scale".format(
                z) for z in range(6)] +
            ["transformer/body/decoder/layer_{}/self_attention/layer_prepostprocess/layer_norm/layer_norm_bias".format(
                z) for z in range(6)] +
            ["transformer/body/decoder/layer_{}/self_attention/multihead_attention/q/kernel".format(z)
             for z in range(6)] +
            ["transformer/body/decoder/layer_{}/self_attention/multihead_attention/k/kernel".format(z)
             for z in range(6)] +
            ["transformer/body/decoder/layer_{}/self_attention/multihead_attention/v/kernel".format(z)
             for z in range(6)] +
            ["transformer/body/decoder/layer_{}/self_attention/multihead_attention/output_transform/kernel".format(
                z) for z in range(6)] +
            ["transformer/body/decoder/layer_{}/encdec_attention/layer_prepostprocess/layer_norm/layer_norm_scale".format(
                    z) for z in range(6)] +
            ["transformer/body/decoder/layer_{}/encdec_attention/layer_prepostprocess/layer_norm/layer_norm_bias".format(
                    z) for z in range(6)] +
            ["transformer/body/decoder/layer_{}/encdec_attention/multihead_attention/q/kernel".format(
                z) for z in range(6)] +
            ["transformer/body/decoder/layer_{}/encdec_attention/multihead_attention/k/kernel".format(
                z) for z in range(6)] +
            ["transformer/body/decoder/layer_{}/encdec_attention/multihead_attention/v/kernel".format(
                z) for z in range(6)] +
            ["transformer/body/decoder/layer_{}/encdec_attention/multihead_attention/output_transform/kernel".format(
                z) for z in range(6)] +
            ["transformer/body/decoder/layer_{}/ffn/layer_prepostprocess/layer_norm/layer_norm_scale".format(
                z) for z in range(6)] +
            ["transformer/body/decoder/layer_{}/ffn/layer_prepostprocess/layer_norm/layer_norm_bias".format(
                z) for z in range(6)] +
            ["transformer/body/decoder/layer_{}/ffn/conv1/kernel".format(z) for z in
             range(6)] +
            ["transformer/body/decoder/layer_{}/ffn/conv1/bias".format(z) for z in range(6)] +
            ["transformer/body/decoder/layer_{}/ffn/conv2/kernel".format(z) for z in
             range(6)] +
            ["transformer/body/decoder/layer_{}/ffn/conv2/bias".format(z) for z in range(6)] +
            ["transformer/body/decoder/layer_prepostprocess/layer_norm/layer_norm_scale"] +
            ["transformer/body/decoder/layer_prepostprocess/layer_norm/layer_norm_bias"] +
            ["transformer/symbol_modality_25000_512/softmax/weights_{}".format(z) for z in
             range(16)])
    write_graph_to_file(_GRAPH_FILE, frozen_graph_def, output_dir)

    return frozen_graph_def


def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


def get_tftrt_name(graph_name, precision_string):
    return "tftrt_{}_{}".format(precision_string.lower(), graph_name)


def get_trt_graph(graph_name, graph_def, precision_mode, output_dir,
                  output_nodes, batch_size=128, workspace_size=2 << 10):
    """Create and save inference graph using the TensorRT library.
    Args:
      graph_name: string, name of the graph to be used for saving.
      graph_def: GraphDef, the Frozen Graph to be converted.
      precision_mode: string, the precision that TensorRT should convert into.
        Options- FP32, FP16, INT8.
      output_dir: string, the path to where files should be written.
      output_nodes: string, the names of the output node that will
        be returned during inference.
      batch_size: int, the number of examples that will be predicted at a time.
      workspace_size: int, size in megabytes that can be used during conversion.
    Returns:
      GraphDef for the TensorRT inference graph.
    """
    trt_graph = trt.create_inference_graph(
        graph_def, output_nodes, max_batch_size=batch_size,
        max_workspace_size_bytes=workspace_size << 20,
        precision_mode=precision_mode)

    write_graph_to_file(graph_name, trt_graph, output_dir)

    return trt_graph


def get_trt_graph_from_calib(graph_name, calib_graph_def, output_dir):
    """Convert a TensorRT graph used for calibration to an inference graph."""
    trt_graph = trt.calib_graph_to_infer_graph(calib_graph_def)
    write_graph_to_file(graph_name, trt_graph, output_dir)
    return trt_graph


################################################################################
# Run the graph in various precision modes.
################################################################################
def get_gpu_config():
    """Share GPU memory between image preprocessing and inference."""
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=_GPU_MEM_FRACTION)
    return tf.ConfigProto(gpu_options=gpu_options)


def run_trt_graph_for_mode(
        graph_name, graph_def, mode, flags):
    """Convert, time, and log the graph at `mode` precision using TensorRT."""
    g_name = get_tftrt_name(graph_name, mode)
    get_trt_graph(
        g_name, graph_def, mode, flags.output_dir, flags.output_nodes,
        flags.batch_size, flags.workspace_size)
    return None


################################################################################
# Run this script
################################################################################
def main(argv):
    parser = TensorRTParser()
    flags = parser.parse_args(args=argv[1:])

    # Load the data.
    # Load the graph def
    if flags.frozen_graph:
        frozen_graph_def = get_frozen_graph(flags.frozen_graph)
    elif flags.savedmodel_dir:
        frozen_graph_def = convert_savedmodel_to_frozen_graph(
            flags.savedmodel_dir, flags.output_dir)
    else:
        raise ValueError(
            "Either a Frozen Graph file or a SavedModel must be provided.")

    # Get a name for saving TensorRT versions of the graph.
    graph_name = os.path.basename(flags.frozen_graph or _GRAPH_FILE)

    # Run inference in all desired modes.

    if flags.fp32:
        mode = "FP32"
        print("Running {} graph".format(mode))
        run_trt_graph_for_mode(graph_name, frozen_graph_def, mode, flags)

    if flags.fp16:
        mode = "FP16"
        print("Running {} graph".format(mode))
        run_trt_graph_for_mode(
            graph_name, frozen_graph_def, mode, flags)


class TensorRTParser(argparse.ArgumentParser):
    """Parser to contain flags for running the TensorRT timers."""

    def __init__(self):
        super(TensorRTParser, self).__init__()

        self.add_argument(
            "--frozen_graph", "-fg", default=None,
            help="[default: %(default)s] The location of a Frozen Graph "
                 "protobuf file that will be used for inference. Note that either "
                 "savedmodel_dir or frozen_graph should be passed in, and "
                 "frozen_graph will take precedence.",
            metavar="<FG>",
        )

        self.add_argument(
            "--savedmodel_dir", "-sd", default=None,
            help="[default: %(default)s] The location of a SavedModel directory "
                 "to be converted into a Frozen Graph. This is equivalent to the "
                 "subdirectory that is created under the directory specified by "
                 "--export_dir when running an Official Model. Note that either "
                 "savedmodel_dir or frozen_graph should be passed in, and "
                 "frozen_graph will take precedence.",
            metavar="<SD>",
        )

        self.add_argument(
            "--output_dir", "-od", default="/tmp",
            help="[default: %(default)s] The location where output files will "
                 "be saved.",
            metavar="<OD>",
        )

        self.add_argument(
            "--output_nodes", "-on", nargs='+',
            default=['DatasetToSingleElement:0',
                     'transformer/strided_slice_10:0',
                     'transformer/strided_slice_11:0'],
            help="[default: %(default)s] The names of the graph output node "
                 "that should be used when retrieving results. Assumed to be a softmax.",
            metavar="<ON>",
        )

        self.add_argument(
            "--batch_size", "-bs", type=int, default=5,
            help="[default: %(default)s] Batch size for inference. If an "
                 "image file is passed, it will be copied batch_size times to "
                 "imitate a batch.",
            metavar="<BS>"
        )

        self.add_argument(
            "--fp32", action="store_true",
            help="[default: %(default)s] If set, benchmark the model with TensorRT "
                 "using fp32 precision."
        )

        self.add_argument(
            "--fp16", action="store_true",
            help="[default: %(default)s] If set, benchmark the model with TensorRT "
                 "using fp16 precision."
        )

        self.add_argument(
            "--workspace_size", "-ws", type=int, default=2 << 10,
            help="[default: %(default)s] Workspace size in megabytes.",
            metavar="<WS>"
        )


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    main(argv=sys.argv)
