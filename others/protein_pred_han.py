import numpy as np
import HAN
import argparse
from keras.layers import LSTM, GRU

import tensorflow as tf

import keras.backend.tensorflow_backend as KTF


parser = argparse.ArgumentParser(description='remover')
parser.add_argument('--time-steps', type=int, default=13,
                    help='time steps for rnn')
parser.add_argument('--n-inputs', type=int, default=20,
                    help='num of inputs')
parser.add_argument('--rnn-units', type=int, default=32,
                    help='# of units for rnn')
parser.add_argument('--dense-units', type=int, default=32,
                    help='# of units for dense')
parser.add_argument('--n-classes', type=int, default=3,
                    help='# of classes')
parser.add_argument('--batch-size', type=int, default=32,
                    help='batch size')
parser.add_argument('--n-epochs', type=int, default=200,
                    help='# of epochs')
parser.add_argument('--batch-normalization', type=bool, default=True,
                    help='whether use batch normalization')
parser.add_argument('--rnn-type', type=str, default="lstm",
                    choices=['gru', 'lstm'],
                    help='the type of rnn')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='the dropout rate for network')
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adagrad', 'sgd', 'rmsprop'],
                    help='optimizer type')
parser.add_argument('--embedding-size', type=int, default=10,
                    help='embedding_length')
parser.add_argument('--vocab-size', type=int, default=30,
                    help='vocab_size')
parser.add_argument('--x-train', type=str, default='X_train.npy',
                    help='path to X train')
parser.add_argument('--y-train', type=str, default='y_train.npy',
                    help='path to y train')
parser.add_argument('--x-eval', type=str, default='X_eval.npy',
                    help='path to X eval')
parser.add_argument('--y-eval', type=str, default='y_eval.npy',
                    help='path to y eval')
parser.add_argument('--x-test', type=str, default='X_test.npy',
                    help='path to X test or X eval')
parser.add_argument('--y-test', type=str, default='y_test.npy',
                    help='path to y test or y eval')
parser.add_argument('--model-dir', type=str, default='./',
                    help='model dir to save the model')
parser.add_argument('--early-stopping', type=int, default=5,
                    help='round for early stopping')
parser.add_argument('--embed-trainable', type=bool, default=True,
                    help='round for early stopping')

args = parser.parse_args()


def main():

    han_network = HAN.HAN(
        args.time_steps,
        args.n_inputs,
        args.embedding_size,
        args.vocab_size,
        num_classes=args.n_classes,
        verbose=1
    )

    han_network.set_hyperparameters(l2_regulizer=1e-4,
                                    rnn=GRU if args.rnn_type == "gru" else LSTM,
                                    dropout_regulizer=args.dropout,
                                    rnn_units=args.rnn_units,
                                    dense_units=args.dense_units,
                                    optimizer=args.optimizer,
                                    embed_trainable=args.embed_trainable,
                                    model_dir=args.model_dir)

    X_train, y_train = np.load(args.x_train), np.load(args.y_train)
    X_eval, y_eval = np.load(args.x_eval), np.load(args.y_eval)
    X_test, y_test = np.load(args.x_test), np.load(args.y_test)
    han_network.train(X_train,
                      y_train,
                      X_eval,
                      y_eval,
                      args.n_epochs,
                      args.batch_size,
                      best_model_path='best_han.h5py',
                      final_model_path='final_han.h5py',
                      )
    han_network.evaluate(X_test, y_test)


if __name__ == '__main__':
    config = tf.ConfigProto()

    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    KTF.set_session(sess)

    main()
