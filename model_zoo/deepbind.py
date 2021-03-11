""" DeepBind model from: 
    paper: Alipanahi et al. Predicting the sequence specificities of DNA-and RNA-binding proteins by deep learning. Nature Biotechnology. 2015 Aug;33(8):831-8.
    url: https://www.nature.com/articles/nbt.3300
"""
from tensorflow import keras


def model(input_shape, output_shape, units=[24, 48], dropout=[0.0, 0.5]):
  
    # input layer
    inputs = keras.layers.Input(shape=input_shape)

    # layer 1
    nn = keras.layers.Conv1D(filters=units[0],
                             kernel_size=23,
                             strides=1,
                             activation='relu',
                             use_bias=True,
                             padding='same',
                             )(inputs)
    nn = keras.layers.GlobalMaxPooling1D()(nn)
    nn = keras.layers.Dropout(dropout[0])(nn)

    # layer 2 
    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(units[1], activation='relu', use_bias=True)(nn)      
    nn = keras.layers.Dropout(dropout[1])(nn)

    # Output layer
    logits = keras.layers.Dense(output_shape, activation='linear', use_bias=True)(nn)
    outputs = keras.layers.Activation('sigmoid')(logits)

    # create keras model
    return keras.Model(inputs=inputs, outputs=outputs)
