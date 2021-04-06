from tensorflow import keras


def model(
    input_shape,
    output_shape,
    activation="relu",
    units=[24, 32, 48, 64, 96],
    dropout=[0.1, 0.2, 0.3, 0.4, 0.5],
):
    """Creates a keras neural network with the architecture shown below. The
    architecture is chosen to promote learning in a distributive way.

    Parameters
    ----------
    input_shape: tuple
        Tuple of size (L,4) where L is the sequence lenght and 4 is the number of 1-hot
        channels. Assumes all sequences have equal length.
    output_shape: int
        Number of output categories.
    activation: str, optional
        A string specifying the type of activation. Example: 'relu', 'exponential', ...
    units: list, optional
        A list of 5 integers that can be used to specify the number of filters.
    dropout: list, optional
        A list of length 5 with probabilities that can be used to set the probabilities
        of dropouts in the main architecture.

    Returns
    -------
    Keras Functional Model

    Example
    -------
    >>> model = cnn_dist_model(
        (200,4), 1 , 'relu', [24, 32, 48, 64, 96 ], [0.1, 0.2, 0.3, 0.4, 0.5] )
    """

    # input layer
    inputs = keras.layers.Input(shape=input_shape)

    # block 1
    nn = keras.layers.Conv1D(
        filters=units[0],
        kernel_size=19,
        padding="same",
        activation=activation,
        kernel_regularizer=keras.regularizers.l2(1e-6),
    )(inputs)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Dropout(dropout[0])(nn)

    # layer 2
    nn = keras.layers.Conv1D(
        filters=units[1],
        kernel_size=7,
        padding="same",
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-6),
    )(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Dropout(dropout[1])(nn)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)

    # layer 3
    nn = keras.layers.Conv1D(
        filters=units[2],
        kernel_size=7,
        padding="valid",
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-6),
    )(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Dropout(dropout[2])(nn)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)

    # layer 4
    nn = keras.layers.Conv1D(
        filters=units[3],
        kernel_size=3,
        padding="valid",
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-6),
    )(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Dropout(dropout[3])(nn)
    nn = keras.layers.MaxPool1D(pool_size=3, strides=3, padding="same")(nn)

    # layer 5
    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(
        units=units[4],
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-6),
    )(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Dropout(dropout[4])(nn)

    # Output layer
    logits = keras.layers.Dense(output_shape, activation="linear", use_bias=True)(nn)
    outputs = keras.layers.Activation("sigmoid")(logits)

    # compile model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
