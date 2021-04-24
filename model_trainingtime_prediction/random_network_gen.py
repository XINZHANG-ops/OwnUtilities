import numpy as np
from keras.models import Sequential
from keras.layers import Dense

class gen_nn:
    def __init__(self,
                 input_dim_lower = 1,
                 input_dim_upper = 1001,
                 output_dim_lower = 1,
                 output_dim_upper = 1001,
                 hidden_layers_num_lower = 5,
                 hidden_layers_num_upper = 100,
                 hidden_layer_size_lower = 1,
                 hidden_layer_size_upper = 1001,
                 activition = 'random',
                 optimizer = 'random',
                 loss = 'random'
                 ):
        self.input_dim_lower = input_dim_lower
        self.input_dim_upper = input_dim_upper
        self.output_dim_lower = output_dim_lower
        self.output_dim_upper = output_dim_upper
        self.hidden_layers_num_lower = hidden_layers_num_lower
        self.hidden_layers_num_upper = hidden_layers_num_upper
        self.hidden_layer_size_lower = hidden_layer_size_lower
        self.hidden_layer_size_upper = hidden_layer_size_upper
        self.activition= activition
        self.optimizer = optimizer
        self.loss = loss
        self.activation_fcts = ['relu', "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "exponential"]
        self.optimizers = ["SGD", "RMSprop", "Adam", "Adadelta", "Adagrad", "Adamax", "Nadam", "Ftrl"]
        self.losses =  ["mae", "mape", "mse", "msle", "poisson", "categorical_crossentropy"]

    @staticmethod
    def build_dense_model(input_dim, output_dim, hidden_layer_sizes, activation_fcts, optimizer, loss):
      model_dense = Sequential()
      if list(hidden_layer_sizes):
        model_dense.add(Dense(hidden_layer_sizes[0], input_dim=input_dim, activation=activation_fcts[0]))
        for index,size in enumerate(hidden_layer_sizes[1:]):
          model_dense.add(Dense(size, activation=activation_fcts[index+1]))
        model_dense.add(Dense(output_dim, activation='softmax'))
      else:
        model_dense.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))
      model_dense.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
      return model_dense

    def generate_model(self):
        self.input_dim = np.random.randint(self.input_dim_lower, self.input_dim_upper)
        self.output_dim = np.random.randint(self.output_dim_lower, self.output_dim_upper)
        self.hidden_layers_num = np.random.randint(self.hidden_layers_num_lower, self.hidden_layers_num_upper)
        self.hidden_layer_sizes = np.random.randint(self.hidden_layer_size_lower, self.hidden_layer_size_upper, self.hidden_layers_num)

        if self.activition == 'random':
            self.activations = np.random.choice(self.activation_fcts,self.hidden_layers_num)
        else:
            self.activations = np.random.choice([self.activition],self.hidden_layers_num)
        if self.optimizer == 'random':
            self.optimizer = np.random.choice(self.optimizers)
        if self.loss == 'random':
            self.loss = np.random.choice(self.losses)

        return  gen_nn.build_dense_model(self.input_dim, self.output_dim, self.hidden_layer_sizes, self.activations, self.optimizer, self.loss)
