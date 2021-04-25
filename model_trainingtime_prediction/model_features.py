# For quick test

"""
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.compile(loss='mse', optimizer='SGD', metrics=['accuracy'])

"""


def get_model_feature(keras_model):
    layer_types = []
    layer_sizes = []
    activations = []
    for layer in keras_model.get_config()['layers']:
        layer_sizes.append(layer['config']['units'])
        activations.append(layer['config']['activation'])
        layer_types.append(layer['class_name'])
    return {'layer_sizes': layer_sizes,
            'activations': activations,
            'layer_types': layer_types}
