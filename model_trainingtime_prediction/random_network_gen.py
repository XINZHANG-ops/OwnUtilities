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
        self.activation_pick = activation
        self.optimizer_pick = optimizer
        self.loss_pick = loss
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
        hidden_layers_num = np.random.randint(self.hidden_layers_num_lower, self.hidden_layers_num_upper)
        hidden_layer_sizes = np.random.randint(self.hidden_layer_size_lower, self.hidden_layer_size_upper, hidden_layers_num)

        if self.activation_pick == 'random':
            activations = np.random.choice(self.activation_fcts,hidden_layers_num)
        else:
            activations = np.random.choice([self.activation_pick],hidden_layers_num)
        if self.optimizer_pick == 'random':
            optimizer = np.random.choice(self.optimizers)
        else:
            optimizer = self.optimizer_pick
        if self.loss_pick == 'random':
            loss = np.random.choice(self.losses)
        else:
            loss =  self.loss_pick

        return {'model': gen_nn.build_dense_model(hidden_layer_sizes, activations, optimizer, loss),
                'layer_sizes': [int(i) for i in hidden_layer_sizes],
                'activations': list(activations),
                'optimizer': optimizer,
                'loss': loss}

