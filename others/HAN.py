from keras import regularizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.layers import Embedding, Input, Dense, GRU, Bidirectional, TimeDistributed, Dropout, Activation
from keras.models import Model
import matplotlib.pyplot as plt
from attention_with_context import AttentionWithContext
from keras.layers.normalization import BatchNormalization
import time
import numpy as np
import os
import sys
from keras.models import load_model


class HAN(object):
    """
    HAN model is implemented here.
    """

    def __init__(self,
                 time_steps,
                 n_inputs,
                 embedding_size,
                 vocab_size,
                 num_classes=None,
                 verbose=0):
        """Initialize the HAN module
        Keyword arguments:
        time_steps -- time_steps
        n_inputs -- feature length
        embedding_size -- size of the embedding vector
        num_classes -- total number of categories.
        validation_split -- train-test split.
        verbose -- how much you want to see.
        """
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.time_steps = time_steps
        self.n_inputs = n_inputs
        self.verbose = verbose
        self.vocab_size = vocab_size
        self.hyperparameters = {
            'l2_regulizer': None,
            'dropout_regulizer': None,
            'rnn': GRU,
            'rnn_units': 32,
            'dense_units': 32,
            'activation': 'softmax',
            'optimizer': 'adam',
            'metrics': ['acc'],
            'loss': 'categorical_crossentropy',
            'early_stopping': 5,
            'embed_trainable': True,
            'model_dir': './'
        }
        self.model = None
        self.history = None
        self.set_model()
        self._trained = False

    def set_hyperparameters(self, **tweaked_instances):
        """Set hyperparameters of HAN model.
        Keywords arguemnts:
        tweaked_instances -- dictionary of all those keys you want to change
        """
        for key, value in tweaked_instances.items():
            if key in self.hyperparameters:
                self.hyperparameters[key] = value
            else:
                raise KeyError(key + ' does not exist in hyperparameters')
            self.set_model()

    def show_hyperparameters(self):
        """To check the values of all the current hyperparameters
        """
        print('Hyperparameter\tCorresponding Value')
        for key, value in self.hyperparameters.items():
            print(key, '\t\t', value)

    def get_model(self):
        """
        Returns the HAN model so that it can be used as a part of pipeline
        """
        return self.model

    @staticmethod
    def get_embedding_layer(vocab_size, embedding_size, n_inputs, trainable=True):
        """
        Returns Embedding layer
        """
        return Embedding(vocab_size, embedding_size, input_length=n_inputs, trainable=trainable)

    @staticmethod
    def _bn_relu(input):
        """Helper to build a BN -> relu block
        """
        norm = BatchNormalization()(input)
        return Activation("relu")(norm)

    def set_model(self):
        """
        Set the HAN model according to the given hyperparameters
        """
        if self.hyperparameters['l2_regulizer'] is None:
            kernel_regularizer = None
        else:
            kernel_regularizer = regularizers.l2(self.hyperparameters['l2_regulizer'])
        if self.hyperparameters['dropout_regulizer'] is None:
            dropout_regularizer = 1
        else:
            dropout_regularizer = self.hyperparameters['dropout_regulizer']
        word_input = Input(shape=(self.n_inputs, ), dtype='float32')
        word_sequences = self.get_embedding_layer(self.vocab_size,
                                                  self.embedding_size,
                                                  self.n_inputs
                                                  )(word_input)
        word_lstm = Bidirectional(
            self.hyperparameters['rnn'](self.hyperparameters['rnn_units'], return_sequences=True,
                                        kernel_regularizer=kernel_regularizer, activation=None))(word_sequences)
        word_activation = self._bn_relu(word_lstm)
        word_dense = TimeDistributed(
            Dense(self.hyperparameters['dense_units'], kernel_regularizer=kernel_regularizer))(word_activation)
        word_att = AttentionWithContext()(word_dense)

        word_encoder = Model(word_input, word_att)

        sent_input = Input(shape=(self.time_steps, self.n_inputs), dtype='float32')
        sent_encoder = TimeDistributed(word_encoder)(sent_input)
        sent_lstm = Bidirectional(self.hyperparameters['rnn'](
            self.hyperparameters['rnn_units'], return_sequences=True, kernel_regularizer=kernel_regularizer,
            activation=None))(
            sent_encoder)
        sent_activation = self._bn_relu(sent_lstm)
        sent_dense = TimeDistributed(
            Dense(self.hyperparameters['dense_units'], kernel_regularizer=kernel_regularizer))(sent_activation)
        sent_att = Dropout(dropout_regularizer)(
            AttentionWithContext()(sent_dense))
        preds = Dense(self.num_classes,  activation='softmax')(sent_att)
        self.model = Model(sent_input, preds)
        self.model.compile(
            loss=self.hyperparameters['loss'], optimizer=self.hyperparameters['optimizer'],
            metrics=self.hyperparameters['metrics'])

    def train(self,
              X_train,
              y_train,
              X_eval,
              y_eval,
              epochs,
              batch_size,
              best_model_path=None,
              final_model_path=None,
              plot_learning_curve=False):
        """Training the model
        epochs -- Total number of epochs
        batch_size -- size of a batch
        best_model_path -- path to save best model i.e. weights with lowest validation score.
        final_model_path -- path to save final model i.e. final weights
        plot_learning_curve -- Want
        to checkout Learning curve
        """
        checkpoint = ModelCheckpoint(os.path.join(self.hyperparameters['model_dir'], best_model_path),
                                     verbose=0, monitor='val_loss', save_best_only=True,
                                     mode='auto')

        early_stopping_callback = EarlyStopping(monitor='val_loss',
                                                min_delta=0,
                                                patience=self.hyperparameters['early_stopping'],
                                                verbose=0,
                                                mode='auto')

        csv_logger = CSVLogger(os.path.join(self.hyperparameters['model_dir'], 'han.csv'))

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

        self.history = self.model.fit(X_train,
                                      y_train,
                                      validation_data=(X_eval, y_eval),
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      verbose=self.verbose,
                                      shuffle=True,
                                      callbacks=[checkpoint, early_stopping_callback, csv_logger, lr_reducer])
        self._trained = True

        if plot_learning_curve:
            self.plot_results()
        if final_model_path is not None:
            self.model.save(os.path.join(self.hyperparameters['model_dir'], final_model_path))

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

    def plot_results(self):
        """
        Plotting learning curve of last trained model.
        """
        # summarize history for accuracy
        plt.subplot(211)
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        # summarize history for loss
        plt.subplot(212)
        plt.plot(self.history.history['val_loss'])
        plt.plot(self.history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        time.sleep(10)
        plt.close()
