import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='print node name')
parser.add_argument('--input', type=str, default=None,
                    help='path to input')
parser.add_argument('--output', type=str, required=True,
                    help='path to output')
args = parser.parse_args()
tf.reset_default_graph()

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(args.input)
    graph_def = tf.get_default_graph().as_graph_def()
    names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    with open(args.output, 'w') as g:
        for n in tf.get_default_graph().as_graph_def().node:
            g.write(n.name)
            g.write('\n')
