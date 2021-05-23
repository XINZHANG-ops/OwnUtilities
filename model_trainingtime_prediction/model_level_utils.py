"""
****************************************
 * @author: Xin Zhang
 * Date: 5/22/21
****************************************
"""
import time
import tensorflow.keras as keras
import pandas as pd
from tqdm import tqdm
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from random import sample
from sklearn.preprocessing import MinMaxScaler
import copy


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.train_start_time = time.time()
        self.epoch_times = []
        self.batch_times = []
        self.epoch_times_detail = []
        self.batch_times_detail = []

    def on_train_end(self, logs={}):
        self.train_end_time = time.time()

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        epoch_time_end = time.time()
        self.epoch_times.append(epoch_time_end - self.epoch_time_start)
        self.epoch_times_detail.append((self.epoch_time_start, epoch_time_end))

    def on_train_batch_begin(self, batch, logs={}):
        self.bacth_time_start = time.time()

    def on_train_batch_end(self, batch, logs={}):
        batch_time_end = time.time()
        self.batch_times.append(batch_time_end - self.bacth_time_start)
        self.batch_times_detail.append((self.bacth_time_start, batch_time_end))

    def relative_by_train_start(self):
        self.epoch_times_detail = np.array(self.epoch_times_detail) - self.train_start_time
        self.batch_times_detail = np.array(self.batch_times_detail) - self.train_start_time
        self.train_end_time = np.array(self.train_end_time) - self.train_start_time


class gen_nn:
    def __init__(
        self,
        hidden_layers_num_lower=5,
        hidden_layers_num_upper=101,
        hidden_layer_size_lower=1,
        hidden_layer_size_upper=1001,
        activation='random',
        optimizer='random',
        loss='random'
    ):
        self.hidden_layers_num_lower = hidden_layers_num_lower
        self.hidden_layers_num_upper = hidden_layers_num_upper
        self.hidden_layer_size_lower = hidden_layer_size_lower
        self.hidden_layer_size_upper = hidden_layer_size_upper
        self.activation_pick = activation
        self.optimizer_pick = optimizer
        self.loss_pick = loss
        self.activation_fcts = [
            'relu', "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu",
            "exponential"
        ]
        self.optimizers = [
            "sgd", "rmsprop", "adam", "adadelta", "adagrad", "adamax", "nadam", "ftrl"
        ]
        self.losses = ["mae", "mape", "mse", "msle", "poisson", "categorical_crossentropy"]

    @staticmethod
    def nothing(x):
        return x

    @staticmethod
    def build_dense_model(layer_sizes, activations, optimizer, loss):
        model_dense = Sequential()
        for index, size in enumerate(layer_sizes):
            model_dense.add(Dense(size, activation=activations[index]))
        model_dense.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        return model_dense

    @staticmethod
    def get_dense_model_features(keras_model):
        layers = [
            layer_info for layer_info in keras_model.get_config()['layers']
            if layer_info['class_name'] == 'Dense'
        ]
        layer_sizes = [l['config']['units'] for l in layers]
        acts = [l['config']['activation'].lower() for l in layers]
        return layer_sizes, acts

    def generate_model(self):
        hidden_layers_num = np.random.randint(
            self.hidden_layers_num_lower, self.hidden_layers_num_upper
        )
        hidden_layer_sizes = np.random.randint(
            self.hidden_layer_size_lower, self.hidden_layer_size_upper, hidden_layers_num
        )

        if self.activation_pick == 'random':
            activations = np.random.choice(self.activation_fcts, hidden_layers_num)
        else:
            activations = np.random.choice([self.activation_pick], hidden_layers_num)
        if self.optimizer_pick == 'random':
            optimizer = np.random.choice(self.optimizers)
        else:
            optimizer = self.optimizer_pick
        if self.loss_pick == 'random':
            loss = np.random.choice(self.losses)
        else:
            loss = self.loss_pick

        return {
            'model': gen_nn.build_dense_model(hidden_layer_sizes, activations, optimizer, loss),
            'layer_sizes': [int(i) for i in hidden_layer_sizes],
            'activations': list(activations),
            'optimizer': optimizer,
            'loss': loss
        }

    def generate_model_configs(self, num_model_data=1000, progress=True):
        model_configs = []
        if progress:
            loop_fun = tqdm
        else:
            loop_fun = gen_nn.nothing
        for i in loop_fun(range(num_model_data)):
            data = self.generate_model()
            del data['model']
            model_configs.append(data)
        return model_configs


class model_train_data:
    def __init__(
        self,
        model_configs,
        batch_sizes=None,
        epochs=None,
        truncate_from=None,
        trials=None,
        batch_strategy='all',
    ):
        """

        @param model_configs:
        @param batch_sizes:
        @param epochs:
        @param truncate_from:
        @param trials:
        @param batch_strategy: str: `random` or `all`, random will random one from batch_sizes for each model,
        'all' will train for all batch_sizes in batch_sizes for each model
        """
        self.model_configs = []
        for info_dict in model_configs:
            d2 = copy.deepcopy(info_dict)
            self.model_configs.append(d2)
        self.batch_sizes = batch_sizes if batch_sizes is not None else [2**i for i in range(1, 9)]
        self.epochs = epochs if epochs is not None else 10
        self.truncate_from = truncate_from if truncate_from is not None else 2
        self.trials = trials if trials is not None else 5
        self.batch_strategy = batch_strategy
        self.activation_fcts = [
            'relu', "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu",
            "exponential"
        ]
        self.optimizers = [
            "sgd", "rmsprop", "adam", "adadelta", "adagrad", "adamax", "nadam", "ftrl"
        ]
        self.losses = ["mae", "mape", "mse", "msle", "poisson", "categorical_crossentropy"]
        self.act_mapping = dict((act, index + 1)
                                for index, act in enumerate(self.activation_fcts))
        self.opt_mapping = dict((opt, index + 1)
                                for index, opt in enumerate(self.optimizers))
        self.loss_mapping = dict((loss, index + 1)
                                 for index, loss in enumerate(self.losses))

    def get_train_data(self, progress=True):
        model_data = []
        model_configs = []
        if progress:
            loop_fun = tqdm
        else:
            loop_fun = gen_nn.nothing
        for info_dict in self.model_configs:
            d2 = copy.deepcopy(info_dict)
            model_configs.append(d2)
        for model_config in loop_fun(model_configs):
            model = gen_nn.build_dense_model(
                layer_sizes=model_config['layer_sizes'],
                activations=model_config['activations'],
                optimizer=model_config['optimizer'],
                loss=model_config['loss']
            )
            if self.batch_strategy == 'all':
                batch_sizes = self.batch_sizes.copy()
            else:
                batch_sizes = sample(self.batch_sizes, 1)
            for batch_size in batch_sizes:
                batch_size_data_batch = []
                batch_size_data_epoch = []
                for _ in range(self.trials):
                    try:
                        input_shape = model.get_config()['layers'][0]['config']['units']
                    except:
                        input_shape = model.get_config(
                        )['layers'][0]['config']['batch_input_shape'][1]
                    out_shape = model.get_config()['layers'][-1]['config']['units']
                    x = np.ones((batch_size, input_shape), dtype=np.float32)
                    y = np.ones((batch_size, out_shape), dtype=np.float32)

                    time_callback = TimeHistory()
                    model.fit(
                        x,
                        y,
                        epochs=self.epochs,
                        batch_size=batch_size,
                        callbacks=[time_callback],
                        verbose=False
                    )
                    times_batch = np.array(time_callback.batch_times)[self.truncate_from:] * 1000
                    times_epoch = np.array(time_callback.epoch_times)[self.truncate_from:] * 1000
                    batch_size_data_batch.extend(times_batch)
                    batch_size_data_epoch.extend(times_epoch)

                model_config[f'batch_size_{batch_size}'] = {
                    'batch_time': np.median(batch_size_data_batch),
                    'epoch_time': np.median(batch_size_data_epoch)
                }
            model_data.append(model_config)
        return model_data

    def convert_config_data(
        self, model_data, layer_num_upper, layer_na_fill=0, act_na_fill=0, min_max_scaler=True
    ):
        data_rows = []
        time_rows = []
        for model_i_data in model_data:
            layer_sizes = model_i_data['layer_sizes'] + [layer_na_fill] * layer_num_upper
            layer_sizes = layer_sizes[:layer_num_upper]
            activations = [self.act_mapping[i]
                           for i in model_i_data['activations']] + [act_na_fill] * layer_num_upper
            activations = activations[:layer_num_upper]
            optimizer = self.opt_mapping[model_i_data['optimizer']]
            loss = self.loss_mapping[model_i_data['loss']]
            batch_names = [k for k in model_i_data.keys() if k.startswith('batch_size')]
            for batch_name in batch_names:
                batch_value = int(batch_name.split('_')[-1])
                batch_time = model_i_data[batch_name]['batch_time']
                epoch_time = model_i_data[batch_name]['epoch_time']
                data_rows.append(layer_sizes + activations + [optimizer, loss, batch_value])
                time_rows.append([batch_time, epoch_time])
        layer_names = [f'layer_{i + 1}_size' for i in range(layer_num_upper)]
        act_names = [f'layer_{i + 1}_activation' for i in range(layer_num_upper)]
        if min_max_scaler:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(np.array(data_rows))
            data_df = pd.DataFrame(
                scaled_data, columns=layer_names + act_names + ['optimizer', 'loss', 'batch_size']
            )
            time_df = pd.DataFrame(time_rows, columns=['batch_time', 'epoch_time'])
            return pd.concat([data_df, time_df], axis=1), scaler
        else:
            data_df = pd.DataFrame(
                data_rows, columns=layer_names + act_names + ['optimizer', 'loss', 'batch_size']
            )
            time_df = pd.DataFrame(time_rows, columns=['batch_time', 'epoch_time'])
            return pd.concat([data_df, time_df], axis=1), None

    def convert_model_data(
        self,
        keras_model,
        layer_num_upper,
        optimizer,
        loss,
        batch_size,
        layer_na_fill=0,
        act_na_fill=0,
        scaler=None
    ):
        layer_sizes, acts = gen_nn.get_dense_model_features(keras_model)
        layer_sizes = layer_sizes + [layer_na_fill] * layer_num_upper
        layer_sizes = layer_sizes[:layer_num_upper]
        acts = [self.act_mapping[i] for i in acts]
        acts = acts + [act_na_fill] * layer_num_upper
        acts = acts[:layer_num_upper]
        optimizer = self.opt_mapping[optimizer]
        loss = self.loss_mapping[loss]
        data = layer_sizes + acts + [optimizer, loss, batch_size]
        layer_names = [f'layer_{i + 1}_size' for i in range(layer_num_upper)]
        act_names = [f'layer_{i + 1}_activation' for i in range(layer_num_upper)]
        if scaler is None:
            return pd.DataFrame([data],
                                columns=layer_names + act_names +
                                ['optimizer', 'loss', 'batch_size'])
        else:
            scaled_data = scaler.transform([data])
            return pd.DataFrame(
                scaled_data, columns=layer_names + act_names + ['optimizer', 'loss', 'batch_size']
            )


def demo():
    data_points = 1000
    gnn = gen_nn(
        hidden_layers_num_lower=5,
        hidden_layers_num_upper=51,
        hidden_layer_size_lower=1,
        hidden_layer_size_upper=501,
        activation='random',
        optimizer='random',
        loss='random'
    )
    model_configs = gnn.generate_model_configs(num_model_data=data_points)
    mtd = model_train_data(
        model_configs,
        batch_sizes=[2**i for i in range(1, 9)],
        epochs=5,
        truncate_from=2,
        trials=2,
        batch_strategy='random',
    )
    model_data = mtd.get_train_data()
    df, scaler = mtd.convert_config_data(
        model_data, layer_num_upper=50, layer_na_fill=0, act_na_fill=0, min_max_scaler=True
    )

    df
