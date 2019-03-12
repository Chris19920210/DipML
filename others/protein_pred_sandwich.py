import sys
from keras.models import load_model
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

import argparse
import os
import resnet_temporal

import tensorflow as tf

import keras.backend.tensorflow_backend as KTF


parser = argparse.ArgumentParser(description='remover')
parser.add_argument('--time-steps', type=int, default=13,
                    help='time steps for rnn')
parser.add_argument('--n-inputs', type=int, default=20,
                    help='num of inputs')
parser.add_argument('--n-classes', type=int, default=3,
                    help='# of classes')
parser.add_argument('--batch-size', type=int, default=32,
                    help='batch size')
parser.add_argument('--n-epochs', type=int, default=200,
                    help='# of epochs')
parser.add_argument('--embedding-size', type=int, default=None,
                    help='embedding size')
parser.add_argument('--vocab-size', type=int, default=None,
                    help='vocab-size')
parser.add_argument('--resnet-type', type=str, default='tiny',
                    choices=['very_tiny', 'tiny', 'small', 'big'],
                    help='# of layers for rnn')
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adagrad', 'sgd', 'rmsprop'],
                    help='optimizer type')
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
parser.add_argument('--early-stopping', type=int, default=30,
                    help='round for early stopping')
parser.add_argument('--checkpoint-epochs', type=int, default=1,
                    help='save the checkpoints epoch')

args = parser.parse_args()


class ProteinSWClassifier(object):
    def __init__(self,
                 time_steps,
                 n_inputs,
                 n_classes,
                 resnet_type,
                 batch_size,
                 n_epochs,
                 optimizer,
                 model_path,
                 checkpoint_epochs,
                 early_stopping,
                 embedding_size=None,
                 vocab_size=None
                 ):
        # Classifier
        self.time_steps = time_steps
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        # Internal
        self._trained = None
        self._optimizer = optimizer
        self._model_path = model_path
        self._checkpoint_epochs = checkpoint_epochs
        self._early_stopping = early_stopping
        self._resnet_type = resnet_type

    def __create_model(self):

        if self._resnet_type == 'tiny':
            self.model = resnet_temporal.TimeResnetBuilder.build_resnet_tiny((self.time_steps,
                                                                              self.n_inputs),
                                                                             self.n_classes,
                                                                             self.embedding_size,
                                                                             self.vocab_size
                                                                             )
        elif self._resnet_type == 'small':
            self.model = resnet_temporal.TimeResnetBuilder.build_resnet_small((self.time_steps,
                                                                               self.n_inputs),
                                                                              self.n_classes,
                                                                              self.embedding_size,
                                                                              self.vocab_size
                                                                              )
        elif self._resnet_type == 'big':
            self.model = resnet_temporal.TimeResnetBuilder.build_resnet_big((self.time_steps,
                                                                             self.n_inputs),
                                                                            self.n_classes,
                                                                            self.embedding_size,
                                                                            self.vocab_size
                                                                            )
        elif self._resnet_type == 'very_tiny':
            self.model = resnet_temporal.TimeResnetBuilder.build_resnet_very_tiny((self.time_steps,
                                                                                   self.n_inputs),
                                                                                  self.n_classes,
                                                                                  self.embedding_size,
                                                                                  self.vocab_size
                                                                                  )
        else:
            errmsg = "[!] Error: no such resnet type"
            print(errmsg, file=sys.stderr)
            sys.exit(0)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self._optimizer,
                           metrics=['accuracy'])

    def train(self,
              X,
              y,
              X_eval,
              y_eval,
              X_ind=None,
              X_val_ind=None,
              save_model=True):
        self.__create_model()

        checkpoint_callback = ModelCheckpoint(filepath=os.path.join(self._model_path,
                                                                    "model-weights.{epoch:02d}-{val_acc:.6f}.hdf5"),
                                              monitor='val_acc',
                                              verbose=1,
                                              period=self._checkpoint_epochs,
                                              save_best_only=True)

        early_stopping_callback = EarlyStopping(monitor='val_loss',
                                                min_delta=0,
                                                patience=self._early_stopping,
                                                verbose=0,
                                                mode='auto')

        csv_logger = CSVLogger(os.path.join(self._model_path, 'resnet_conv1d_{0}.csv'.format(self._resnet_type)))

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

        if self.embedding_size is not None and self.vocab_size is not None:
            self.model.fit([X, X_ind],
                           y,
                           batch_size=self.batch_size,
                           epochs=self.n_epochs,
                           shuffle=True,
                           validation_data=[[X_eval, X_val_ind], y_eval],
                           callbacks=[checkpoint_callback, early_stopping_callback, csv_logger, lr_reducer]
                           )
        else:
            self.model.fit(X,
                           y,
                           batch_size=self.batch_size,
                           epochs=self.n_epochs,
                           shuffle=True,
                           validation_data=(X_eval, y_eval),
                           callbacks=[checkpoint_callback, early_stopping_callback, csv_logger, lr_reducer]
                           )

        self._trained = True

        if save_model:
            self.model.save(os.path.join(self._model_path, "rnn-model.hdf5"))

    def evaluate(self, X_eval, y_eval, model=None):
        if not self._trained and model is None:
            errmsg = "[!] Error: classifier wasn't trained or classifier path is not precised."
            print(errmsg, file=sys.stderr)
            sys.exit(0)

        model = load_model(model) if model else self.model
        test_loss = model.evaluate(X_eval, y_eval)
        print(test_loss)

    def inference(self, X_test, model, path_to_result):
        if not self._trained and model is None:
            errmsg = "[!] Error: classifier wasn't trained or classifier path is not precised."
            print(errmsg, file=sys.stderr)
            sys.exit(0)
        model = load_model(model) if model else self.model
        y_pred = model.predict(X_test)
        np.save(path_to_result, y_pred)


if __name__ == "__main__":
    config = tf.ConfigProto()

    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    KTF.set_session(sess)

    classifier = ProteinSWClassifier(args.time_steps,
                                     args.n_inputs,
                                     args.n_classes,
                                     args.resnet_type,
                                     args.batch_size,
                                     args.n_epochs,
                                     args.optimizer,
                                     args.model_dir,
                                     args.checkpoint_epochs,
                                     args.early_stopping,
                                     args.embedding_size,
                                     args.vocab_size
                                     )
    X_train, y_train = np.load(args.x_train), np.load(args.y_train)
    X_eval, y_eval = np.load(args.x_eval), np.load(args.y_eval)
    #X_test, y_test = np.load(args.x_test), np.load(args.y_test)
    # subtract mean and normalize
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_eval -= mean_image
    #X_test -= mean_image
    X_train /= 29
    X_eval /= 29
    #X_test /= 29

    if args.embedding_size is not None and args.vocab_size is not None:
        X_ind = np.load(args.x_train[:args.x_train.find(".")] + "_ind" + args.x_train[args.x_train.find("."):])
        X_eval_ind = np.load(args.x_eval[:args.x_eval.find(".")] + "_ind" + args.x_eval[args.x_eval.find("."):])
        classifier.train(X_train, y_train, X_eval, y_eval, X_ind, X_eval_ind, save_model=True)
    else:
        classifier.train(X_train, y_train, X_eval, y_eval, save_model=True)
