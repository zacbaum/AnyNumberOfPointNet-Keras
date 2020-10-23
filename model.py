import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Input,
    Dense,
    Reshape,
    BatchNormalization,
    Dropout,
    Flatten,
)
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
import numpy as np


class MatMul(Layer):
    def __init__(self, **kwargs):
        super(MatMul, self).__init__(**kwargs)

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list):
            raise ValueError("`MatMul` layer should be called " "on a list of inputs")
        if len(input_shape) != 2:
            raise ValueError(
                "The input of `MatMul` layer should be a list containing 2 elements"
            )

        if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
            raise ValueError("The dimensions of each element of inputs should be 3")

        if input_shape[0][-1] != input_shape[1][1]:
            raise ValueError(
                "The last dimension of inputs[0] should match the dimension 1 of inputs[1]"
            )

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError(
                "A `MatMul` layer should be called " "on a list of inputs."
            )
        import tensorflow as tf

        return tf.matmul(inputs[0], inputs[1])

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0][0], input_shape[0][1], input_shape[1][-1]]
        return tuple(output_shape)


def PointNet_cls(classes, input_len=2048, dimensions=3):

    input_points = Input(shape=(input_len, dimensions))

    # input transformation net
    x = Conv1D(64, 1, activation="relu")(input_points)
    x = BatchNormalization()(x)
    x = Conv1D(128, 1, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv1D(1024, 1, activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=input_len)(x)

    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)

    x = Dense(
        dimensions * dimensions,
        weights=[
            np.zeros([256, dimensions * dimensions]),
            np.eye(dimensions).flatten().astype(np.float32),
        ],
    )(x)
    input_T = Reshape((dimensions, dimensions))(x)

    # forward net
    g = MatMul()([input_points, input_T])
    g = Conv1D(64, 1, activation="relu")(g)
    g = BatchNormalization()(g)
    g = Conv1D(64, 1, activation="relu")(g)
    g = BatchNormalization()(g)

    # feature transform net
    f = Conv1D(64, 1, activation="relu")(g)
    f = BatchNormalization()(f)
    f = Conv1D(128, 1, activation="relu")(f)
    f = BatchNormalization()(f)
    f = Conv1D(1024, 1, activation="relu")(f)
    f = BatchNormalization()(f)
    f = MaxPooling1D(pool_size=input_len)(f)
    f = Dense(512, activation="relu")(f)
    f = BatchNormalization()(f)
    f = Dense(256, activation="relu")(f)
    f = BatchNormalization()(f)
    f = Dense(
        64 * 64,
        weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)],
    )(f)
    feature_T = Reshape((64, 64))(f)

    # forward net
    g = MatMul()([g, feature_T])
    g = Conv1D(64, 1, activation="relu")(g)
    g = BatchNormalization()(g)
    g = Conv1D(128, 1, activation="relu")(g)
    g = BatchNormalization()(g)
    g = Conv1D(1024, 1, activation="relu")(g)
    g = BatchNormalization()(g)

    # global features
    global_feature = MaxPooling1D(pool_size=input_len)(g)

    # classifier
    c = Dense(512, activation="relu")(global_feature)
    c = BatchNormalization()(c)
    c = Dropout(0.5)(c)
    c = Dense(256, activation="relu")(c)
    c = BatchNormalization()(c)
    c = Dropout(0.5)(c)
    c = Dense(classes, activation="softmax")(c)
    prediction = Flatten()(c)

    model = Model(inputs=input_points, outputs=prediction, name="PointNet")

    return model
