import sys
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Dropout, Activation
from keras.models import load_model
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,CSVLogger
from keras.layers.normalization import BatchNormalization
from keras.layers import Bidirectional
import argparse
import os

parser = argparse.ArgumentParser(description='remover')
parser.add_argument('--time-steps', type=int, default=13,
                    help='time steps for rnn')
parser.add_argument('--n-inputs', type=int, default=20,
                    help='num of inputs')
parser.add_argument('--n-units', type=int, default=32,
                    help='# of units for rnn')
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
parser.add_argument('--layers', type=int, default=2,
                    help='# of layers for rnn')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='the dropout rate for network')
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
parser.add_argument('--early-stopping', type=int, default=5,
                    help='round for early stopping')
parser.add_argument('--checkpoint-epochs', type=int, default=5,
                    help='save the checkpoints epoch')

args = parser.parse_args()


class ProteinRNNClassifier(object):
    def __init__(self,
                 time_steps,
                 n_units,
                 n_inputs,
                 n_classes,
                 batch_size,
                 n_epochs,
                 rnn,
                 optimizer,
                 layers,
                 dropout,
                 model_path,
                 batch_normal,
                 checkpoint_epochs,
                 early_stopping
                 ):
        # Classifier
        self.time_steps = time_steps  # timesteps to unroll
        self.n_units = n_units  # hidden LSTM units
        self.n_inputs = n_inputs  # rows of 28 pixels (an mnist img is 28x28)
        self.n_classes = n_classes  # mnist classes/labels (0-9)
        self.batch_size = batch_size  # Size of each batch
        self.n_epochs = n_epochs
        self.rnn = rnn
        # Internal
        self._trained = None
        self._optimizer = optimizer
        self._layers = layers
        self._dropout = dropout
        self._model_path = model_path
        self._batch_normal = batch_normal
        self._checkpoint_epochs = checkpoint_epochs
        self._early_stopping = early_stopping

    def __create_model(self):
        self.model = Sequential()
        if self._batch_normal:
            if self._layers > 1:
                self.model.add(Bidirectional(self.rnn(self.n_units, activation=None, return_sequences=True),
                                             input_shape=(self.time_steps, self.n_inputs)))
                self.model.add(BatchNormalization())
                self.model.add(Dropout(self._dropout))
                self.model.add(Activation('tanh'))
            else:
                self.model.add(Bidirectional(self.rnn(self.n_units, activation=None),
                                             input_shape=(self.time_steps, self.n_inputs)))
                self.model.add(BatchNormalization())
                self.model.add(Dropout(self._dropout))
                self.model.add(Activation('tanh'))

            if self._layers > 2:
                for _ in range(self._layers - 2):
                    self.model.add(Bidirectional(self.rnn(self.n_units, activation=None, return_sequences=True)))
                    self.model.add(BatchNormalization())
                    self.model.add(Dropout(self._dropout))
                    self.model.add(Activation('tanh'))

            if self._layers > 1:
                self.model.add(Bidirectional(self.rnn(self.n_units, activation=None)))
                self.model.add(BatchNormalization())
                self.model.add(Dropout(self._dropout))
                self.model.add(Activation('tanh'))
        else:
            if self._layers > 1:
                self.model.add(Bidirectional(self.rnn(self.n_units, return_sequences=True),
                                             input_shape=(self.time_steps, self.n_inputs)))
                self.model.add(Dropout(self._dropout))
            else:
                self.model.add(Bidirectional(self.rnn(self.n_units),
                                             input_shape=(self.time_steps, self.n_inputs)))
                self.model.add(Dropout(self._dropout))

            if self._layers > 2:
                for _ in range(self._layers - 2):
                    self.model.add(Bidirectional(self.rnn(self.n_units, return_sequences=True)))
                    self.model.add(Dropout(self._dropout))
            if self._layers > 1:
                self.model.add(Bidirectional(self.rnn(self.n_units)))
                self.model.add(Dropout(self._dropout))

        self.model.add(Dense(self.n_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self._optimizer,
                           metrics=['accuracy'])

    def train(self, X, y, X_eval, y_eval, save_model):
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

        csv_logger = CSVLogger('rnn_{0}.csv'.format(self.rnn.__name__))

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

        self.model.fit(X, y,
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
        if self._trained == False and model == None:
            errmsg = "[!] Error: classifier wasn't trained or classifier path is not precised."
            print(errmsg, file=sys.stderr)
            sys.exit(0)

        model = load_model(model) if model else self.model
        test_loss = model.evaluate(X_eval, y_eval)
        print(test_loss)

    def inference(self, X_test, model, path_to_result):
        if self._trained == False and model == None:
            errmsg = "[!] Error: classifier wasn't trained or classifier path is not precised."
            print(errmsg, file=sys.stderr)
            sys.exit(0)
        model = load_model(model) if model else self.model
        y_pred = model.predict(X_test)
        np.save(path_to_result, y_pred)


if __name__ == "__main__":
    rnn = None
    if args.rnn_type == 'lstm':
        rnn = LSTM
    elif args.rnn_type == 'gru':
        rnn = GRU

    classifier = ProteinRNNClassifier(args.time_steps,
                                      args.n_units,
                                      args.n_inputs,
                                      args.n_classes,
                                      args.batch_size,
                                      args.n_epochs,
                                      rnn,
                                      args.optimizer,
                                      args.layers,
                                      args.dropout,
                                      args.model_dir,
                                      args.batch_normalization,
                                      args.checkpoint_epochs,
                                      args.early_stopping
                                      )
    X_train, y_train = np.load(args.x_train), np.load(args.y_train)
    X_eval, y_eval = np.load(args.x_eval), np.load(args.y_eval)
    X_test, y_test = np.load(args.x_test), np.load(args.y_test)
    classifier.train(X_train, y_train, X_eval, y_eval, save_model=True)
    classifier.evaluate(X_test, y_test)
