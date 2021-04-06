from tensorflow import keras


def model(
    input_shape,
    output_shape,
    kernel_size=19,
    activation='relu',
    pool_size=10,
    units=[96, 256],
    dropout=[0.1, 0.2, 0.5],
    dilated=True,
    classification=True,
):
    def residual_block(input_layer, filter_size, activation="relu", dilated=False):

        if dilated:
            factor = [2, 4, 8]
        else:
            factor = [1]
        num_filters = input_layer.shape.as_list()[-1]

        nn = keras.layers.Conv1D(
            filters=num_filters,
            kernel_size=filter_size,
            activation=None,
            use_bias=False,
            padding="same",
            dilation_rate=1,
        )(input_layer)
        nn = keras.layers.BatchNormalization()(nn)
        for f in factor:
            nn = keras.layers.Activation("relu")(nn)
            nn = keras.layers.Dropout(0.1)(nn)
            nn = keras.layers.Conv1D(
                filters=num_filters,
                kernel_size=filter_size,
                strides=1,
                activation=None,
                use_bias=False,
                padding="same",
                dilation_rate=f,
            )(nn)
            nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.add([input_layer, nn])
        return keras.layers.Activation(activation)(nn)

    # input layer
    inputs = keras.layers.Input(shape=input_shape)

    # layer 1
    nn = keras.layers.Conv1D(
        filters=units[0],
        kernel_size=kernel_size,
        strides=1,
        activation=None,
        use_bias=False,
        padding="same",
    )(inputs)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(activation)(nn)
    nn = keras.layers.Dropout(dropout[0])(nn)

    # dilated residual block
    nn = residual_block(nn, filter_size=3, dilated=dilated)

    # average pooling
    nn = keras.layers.AveragePooling1D(pool_size=pool_size)(nn)
    nn = keras.layers.Dropout(dropout[1])(nn)

    # Fully-connected NN
    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(units[2], activation=None, use_bias=False)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation("relu")(nn)
    nn = keras.layers.Dropout(dropout[2])(nn)

    # output layer
    outputs = keras.layers.Dense(output_shape, activation=None, use_bias=True)(nn)

    if classification:
        outputs = keras.layers.Activation("sigmoid")(outputs)

    return keras.Model(inputs=inputs, outputs=outputs)
