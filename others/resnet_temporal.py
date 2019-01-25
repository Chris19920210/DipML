from __future__ import division

import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    GRU,
    LSTM,
    Dense,
    Bidirectional,
    TimeDistributed
)
from keras.layers.convolutional import (
    Conv1D,
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from attention_with_context import AttentionWithContext
from HAN import HAN
from keras.layers.merge import concatenate


def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization()(input)
    return Activation("relu")(norm)


def gru_bn_relu(**rnn_params):
    units = rnn_params["units"]
    return_sequences = rnn_params["return_sequences"]
    kernel_initializer = rnn_params.setdefault("kernel_initializer", "he_normal")
    kernel_regularizer = rnn_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        rnn = Bidirectional(GRU(units=units,
                            return_sequences=return_sequences,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer,
                            activation=None))(input)
        return _bn_relu(rnn)
    return f


def lstm_bn_relu(**rnn_params):
    units = rnn_params["units"]
    return_sequences = rnn_params["return_sequences"]
    kernel_initializer = rnn_params.setdefault("kernel_initializer", "he_normal")
    kernel_regularizer = rnn_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        rnn = Bidirectional(LSTM(units=units,
                            return_sequences=return_sequences,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer,
                            activation=None))(input)
        return _bn_relu(rnn)
    return f


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", 1)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv1D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", 1)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv1D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[TIME_AXIS] / residual_shape[TIME_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or not equal_channels:
        shortcut = Conv1D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=1,
                          strides=stride_width,
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = 1
            # if i == 0 and not is_first_layer:
            #     input_shape = K.int_shape(input)
            #     height = input_shape[TIME_AXIS]
            #     if height <= 2:
            #         init_strides = 1
            #     else:
            #         init_strides = 1
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters, init_strides=1, is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv1D(filters=filters, kernel_size=3,
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=3,
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=3)(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(filters, init_strides=1, is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of filters
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1 = Conv1D(filters=filters, kernel_size=1,
                            strides=init_strides,
                            padding="same",
                            kernel_initializer="he_normal",
                            kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1 = _bn_relu_conv(filters=filters, kernel_size=1,
                                   strides=init_strides)(input)

        conv_3 = _bn_relu_conv(filters=filters, kernel_size=3)(conv_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=1)(conv_3)
        return _shortcut(input, residual)

    return f


def _handle_dim_ordering():
    global TIME_AXIS
    global CHANNEL_AXIS
    TIME_AXIS = 1
    CHANNEL_AXIS = 2


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class TimeResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, rnn_fn, cnn_repetitions, filters, rnn_units,
              dense_units=50, rnn_layers=None, embedding_size=None, vocab_size=None):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape: The input shape in the form (time_steps, channels)
            num_outputs: The number of outputs at final softmax layer
            rnn_fn: The rnn
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            cnn_repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved (not necessarily)
            filters: filter for start.
            rnn_units: rnn units
            dense_units: dense units
            rnn_layers: num of layers, tuple (before cnn layers, after cnn layers)
            embedding_size:
            vocab_size:

        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 2:
            raise Exception("Input shape should be a tuple (time_steps, channels)")

        input = Input(shape=input_shape)
        block_fn = _get_block(block_fn)
        # Load function from str if needed.
        if rnn_layers is None:
            conv1 = _conv_bn_relu(filters=filters, kernel_size=3, strides=1)(input)
            #pool1 = MaxPooling1D(pool_size=2, strides=1, padding="same")(conv1)

            block = conv1
            filters = filters
            for i, r in enumerate(cnn_repetitions):
                block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
                filters *= 2

            # Last activation
            block = _bn_relu(block)

            # Classifier block
            block = TimeDistributed(
                Dense(dense_units, kernel_regularizer=l2(1e-4)))(block)

            block = AttentionWithContext()(block)

            dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                          activation="softmax")(block)

            model = Model(inputs=input, outputs=dense)
            return model
        else:
            rnn_fn = _get_block(rnn_fn)
            bottom = rnn_layers[0]
            up = rnn_layers[1]
            rnn = input
            for _ in range(bottom):
                rnn = rnn_fn(units=rnn_units, return_sequences=True)(rnn)
            conv1 = _conv_bn_relu(filters=filters, kernel_size=3, strides=1)(rnn)

            block = conv1
            filters = filters
            for i, r in enumerate(cnn_repetitions):
                block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
                filters *= 2

            # Last activation
            block = _bn_relu(block)

            for _ in range(up):
                block = rnn_fn(units=rnn_units, return_sequences=True)(block)

            rnn = TimeDistributed(
                Dense(dense_units, kernel_regularizer=l2(1e-4)))(block)

            rnn = AttentionWithContext()(rnn)

            if embedding_size is not None and vocab_size is not None:
                han = HAN(input_shape[0],
                          input_shape[1],
                          embedding_size,
                          vocab_size,
                          num_classes=num_outputs)
                sent_input, sent_att = han.body()
                rnn = concatenate([rnn, sent_att])
                dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                              activation="softmax")(rnn)
                model = Model(inputs=[input, sent_input], outputs=dense)
                return model
            else:
                dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                              activation="softmax")(rnn)
                model = Model(inputs=input, outputs=dense)
                return model

    @staticmethod
    def build_resnet_very_tiny(input_shape, num_outputs, embedding_size=None, vocab_size=None):
        return TimeResnetBuilder.build(input_shape, num_outputs, basic_block, gru_bn_relu, [2], 16, 32, 32, [0, 0],
                                       embedding_size, vocab_size)

    @staticmethod
    def build_resnet_tiny(input_shape, num_outputs, embedding_size=None, vocab_size=None):
        return TimeResnetBuilder.build(input_shape, num_outputs, basic_block, gru_bn_relu, [2, 2], 16, 32, 32, [0, 1],
                                       embedding_size, vocab_size)

    @staticmethod
    def build_resnet_small(input_shape, num_outputs, embedding_size=None, vocab_size=None):
        return TimeResnetBuilder.build(input_shape, num_outputs, basic_block, gru_bn_relu, [2, 2, 2], 16, 32, 32,
                                       [0, 1], embedding_size, vocab_size)

    @staticmethod
    def build_resnet_big(input_shape, num_outputs, embedding_size=None, vocab_size=None):
        return TimeResnetBuilder.build(input_shape, num_outputs, basic_block, gru_bn_relu, [2, 2, 2, 2], 16, 32,
                                       32, [0, 1], embedding_size, vocab_size)
