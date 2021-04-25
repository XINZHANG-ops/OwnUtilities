import numpy as np
from keras.models import Sequential
from keras.layers import Dense

class gen_nn:
    def __init__(self,
                 hidden_layers_num_lower = 5,
                 hidden_layers_num_upper = 100,
                 hidden_layer_size_lower = 1,
                 hidden_layer_size_upper = 1001,
                 activation ='random',
                 optimizer = 'random',
                 loss = 'random'
                 ):
        self.hidden_layers_num_lower = hidden_layers_num_lower
        self.hidden_layers_num_upper = hidden_layers_num_upper
        self.hidden_layer_size_lower = hidden_layer_size_lower
        self.hidden_layer_size_upper = hidden_layer_size_upper
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.activation_fcts = ['relu', "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "exponential"]
        self.optimizers = ["SGD", "RMSprop", "Adam", "Adadelta", "Adagrad", "Adamax", "Nadam", "Ftrl"]
        self.losses =  ["mae", "mape", "mse", "msle", "poisson", "categorical_crossentropy"]

    @staticmethod
    def build_dense_model(hidden_layer_sizes, activation_fcts, optimizer, loss):
        model_dense = Sequential()
        for index,size in enumerate(hidden_layer_sizes):
          model_dense.add(Dense(size, activation=activation_fcts[index]))
        model_dense.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        return model_dense

    def generate_model(self):
        self.hidden_layers_num = np.random.randint(self.hidden_layers_num_lower, self.hidden_layers_num_upper)
        self.hidden_layer_sizes = np.random.randint(self.hidden_layer_size_lower, self.hidden_layer_size_upper, self.hidden_layers_num)

        if self.activation == 'random':
            self.activations = np.random.choice(self.activation_fcts,self.hidden_layers_num)
        else:
            self.activations = np.random.choice([self.activation], self.hidden_layers_num)
        if self.optimizer == 'random':
            self.optimizer = np.random.choice(self.optimizers)
        if self.loss == 'random':
            self.loss = np.random.choice(self.losses)

        return {'model': gen_nn.build_dense_model(self.hidden_layer_sizes, self.activations, self.optimizer, self.loss),
                'layer_sizes': self.hidden_layer_sizes,
                'activations': self.activations,
                'optimizer': self.optimizer,
                'loss': self.loss}

