""" DeepBind model from:
    paper: Alipanahi et al. Predicting the sequence specificities of DNA-and
    RNA-binding proteins by deep learning. Nature Biotechnology. 2015 Aug;33(8):831-8.
    url: https://www.nature.com/articles/nbt.3300
"""
from tensorflow import keras


def model(input_shape, output_shape, activation='relu', units=[24, 48], dropout=[0.1, 0.5]):


    """
    Creates a keras neural network model similar to the original DeepBind architecture. The difference is that it allows for custom specification of the activation of the first convolutional layer. 


    Parameters
    ----------
    input_shape: tuple
        Tuple of size (L,4) where L is the sequence lenght and 4 is the number of 1-hot channels. Assumes all sequences have equal length. 

    output_shape: int
        Number of output categories.

    activation: str
        A string specifying the type of activation. Example: 'relu', 'exponential', ...

    units: list
        Optional parameter. A list of shape [int, int] that can be used to specify the number of filters. It provide more external control of the architecture.  

    dropout: list
        Optional parameter. A list of the form [probability, probability] that can be used to externally control the probabilities of dropouts in the main architecture.   


    Returns
    ----------
    A keras model instance. 


    Example
    -----------
    model = deepbind_custom( (200,4), 1 , 'relu', [24,48], [0.1, 0.5] )

    """

    # input layer
    inputs = keras.layers.Input(shape=input_shape)

    # layer 1
    nn = keras.layers.Conv1D(
        filters=units[0],
        kernel_size=23,
        strides=1,
        activation=activation,
        use_bias=True,
        padding="same",
    )(inputs)
    nn = keras.layers.GlobalMaxPooling1D()(nn)
    nn = keras.layers.Dropout(dropout[0])(nn)

    # layer 2
    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(units[1], activation="relu", use_bias=True)(nn)
    nn = keras.layers.Dropout(dropout[1])(nn)

    # Output layer
    logits = keras.layers.Dense(output_shape, activation="linear", use_bias=True)(nn)
    outputs = keras.layers.Activation("sigmoid")(logits)

    # create keras model
    return keras.Model(inputs=inputs, outputs=outputs)
