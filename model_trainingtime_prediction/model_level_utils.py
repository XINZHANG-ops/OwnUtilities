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
        batch_strategy='random',
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
        self.act_mapping = dict((act, index + 1) for index, act in enumerate(self.activation_fcts))
        self.opt_mapping = dict((opt, index + 1) for index, opt in enumerate(self.optimizers))
        self.loss_mapping = dict((loss, index + 1) for index, loss in enumerate(self.losses))

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
                    times_batch = np.array(time_callback.batch_times) * 1000
                    times_epoch = np.array(time_callback.epoch_times) * 1000
                    batch_size_data_batch.extend(times_batch)
                    batch_size_data_epoch.extend(times_epoch)

                batch_times_truncated = batch_size_data_batch[self.truncate_from:]
                epoch_times_trancuted = batch_size_data_epoch[self.truncate_from:]
                recovered_time = [
                    np.median(batch_times_truncated)
                ] * self.truncate_from + batch_times_truncated

                model_config[f'batch_size_{batch_size}'] = {
                    'batch_time': np.median(batch_times_truncated),
                    'epoch_time': np.median(epoch_times_trancuted),
                    'setup_time': np.sum(batch_size_data_batch) - sum(recovered_time)
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
                setup_time = model_i_data[batch_name]['setup_time']
                data_rows.append(layer_sizes + activations + [optimizer, loss, batch_value])
                time_rows.append([batch_time, epoch_time, setup_time])

        layer_names = [f'layer_{i + 1}_size' for i in range(layer_num_upper)]
        act_names = [f'layer_{i + 1}_activation' for i in range(layer_num_upper)]
        if min_max_scaler:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(np.array(data_rows))
            data_df = pd.DataFrame(
                scaled_data, columns=layer_names + act_names + ['optimizer', 'loss', 'batch_size']
            )
            time_df = pd.DataFrame(time_rows, columns=['batch_time', 'epoch_time', 'setup_time'])
            return pd.concat([data_df, time_df], axis=1), scaler
        else:
            data_df = pd.DataFrame(
                data_rows, columns=layer_names + act_names + ['optimizer', 'loss', 'batch_size']
            )
            time_df = pd.DataFrame(time_rows, columns=['batch_time', 'epoch_time', 'setup_time'])
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
    import random
    import matplotlib.pyplot as plt
    # whole pipeline demo

    # generate model configurations as data points
    data_points = 1000
    gnn = gen_nn(
        hidden_layers_num_lower=1,
        hidden_layers_num_upper=51,
        hidden_layer_size_lower=1,
        hidden_layer_size_upper=1001,
        activation='random',
        optimizer='random',
        loss='random'
    )
    model_configs = gnn.generate_model_configs(num_model_data=data_points)

    # train generated model configurations to get training time
    mtd = model_train_data(
        model_configs,
        batch_sizes=[2**i for i in range(1, 9)],
        epochs=5,
        truncate_from=1,
        trials=2,
        batch_strategy='random',
    )
    model_data = mtd.get_train_data()

    # convert raw data as dataframe and scaler
    df, scaler = mtd.convert_config_data(
        model_data, layer_num_upper=50, layer_na_fill=0, act_na_fill=0, min_max_scaler=True
    )

    # use data to train a ML model
    test_ratio = 0.2
    df_index = df.index.tolist()
    np.random.shuffle(df_index)

    middle_index = int(df.shape[0] * test_ratio)
    test_idx = df_index[:middle_index]
    train_idx = df_index[middle_index:]

    df_train = df.iloc[train_idx]
    df_test = df.iloc[test_idx]

    # we need to train 2 models, one to predict batch runtime, one to predict setup time
    # combine both will be the true training time of a model
    feature_cols = df.columns.tolist()[:-3]
    target_col = 'batch_time'
    setup_col = 'setup_time'

    x_train = df_train[feature_cols].to_numpy()
    y_batch_train = np.array(df_train[target_col].tolist())
    y_setup_train = np.array(df_train[setup_col].tolist())

    x_test = df_test[feature_cols].to_numpy()
    y_batch_test = np.array(df_test[target_col].tolist())
    y_setup_test = np.array(df_test[setup_col].tolist())

    # build a regular dense model for batch time prediction
    from keras.models import Sequential
    from keras.layers import Dense

    batch_model = Sequential()
    batch_model.add(
        Dense(200, input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu')
    )
    batch_model.add(Dense(200, kernel_initializer='normal', activation='relu'))
    batch_model.add(Dense(200, kernel_initializer='normal', activation='relu'))
    batch_model.add(Dense(200, kernel_initializer='normal', activation='relu'))
    batch_model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    batch_model.compile(loss='mean_squared_error', optimizer='adam')

    history_batch = batch_model.fit(
        x_train,
        y_batch_train,
        batch_size=16,
        epochs=50,
        validation_data=(x_test, y_batch_test),
        verbose=True
    )

    # summarize history for loss
    plt.plot(history_batch.history['loss'])
    plt.plot(history_batch.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # plot predictions vs true for batch model
    batch_y_pred = batch_model.predict(x_test)
    batch_y_pred = batch_y_pred.reshape(batch_y_pred.shape[0], )
    plt.scatter(batch_y_pred, y_batch_test)
    plt.show()

    # build a dense model for setup time prediction
    setup_model = Sequential()
    setup_model.add(
        Dense(200, input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu')
    )
    setup_model.add(Dense(200, kernel_initializer='normal', activation='relu'))
    setup_model.add(Dense(200, kernel_initializer='normal', activation='relu'))
    setup_model.add(Dense(200, kernel_initializer='normal', activation='relu'))
    setup_model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    setup_model.compile(loss='mean_squared_error', optimizer='adam')
    history_setup = setup_model.fit(
        x_train,
        y_setup_train,
        batch_size=16,
        epochs=45,
        validation_data=(x_test, y_setup_test),
        verbose=True
    )

    # summarize history for loss
    plt.plot(history_setup.history['loss'])
    plt.plot(history_setup.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # plot predictions vs true for setup time model
    setup_y_pred = setup_model.predict(x_test)
    setup_y_pred = setup_y_pred.reshape(setup_y_pred.shape[0], )
    plt.scatter(setup_y_pred, y_setup_test)
    plt.show()

    # validate on a real case
    val_data_points = 100
    val_genn = gen_nn(
        hidden_layers_num_lower=50,
        hidden_layers_num_upper=51,
        hidden_layer_size_lower=1,
        hidden_layer_size_upper=1001,
        activation='random',
        optimizer='random',
        loss='random'
    )
    val_model_configs = val_genn.generate_model_configs(num_model_data=val_data_points)

    # collect all info during training
    real_time_process_first_batchs = []
    real_time_batchs = []
    real_time_epochs = []
    real_time_start_ends = []
    y_val_preds_batch = []
    y_val_preds_setup = []
    batch_sizes_collect = []
    epochs_collect = []
    data_points_collect = []

    mtd_val = model_train_data([])
    for m_config in tqdm(val_model_configs):
        # here we consider changeable data size and epoch
        batch_size_val = random.sample(mtd_val.batch_sizes, 1)[0]
        epochs_val = random.sample([2, 3, 4, 5], 1)[0]
        data_size_val = random.sample([5000, 10000, 15000, 1000], 1)[0]
        data_points_collect.append(data_size_val)
        batch_sizes_collect.append(batch_size_val)
        epochs_collect.append(epochs_val)

        model_val = gen_nn.build_dense_model(
            layer_sizes=m_config['layer_sizes'],
            activations=m_config['activations'],
            optimizer=m_config['optimizer'],
            loss=m_config['loss']
        )

        try:
            input_shape = model_val.get_config()['layers'][0]['config']['units']
        except:
            input_shape = model_val.get_config()['layers'][0]['config']['batch_input_shape'][1]
        out_shape = model_val.get_config()['layers'][-1]['config']['units']
        x = np.ones((data_size_val, input_shape), dtype=np.float32)
        y = np.ones((data_size_val, out_shape), dtype=np.float32)

        time_callback = TimeHistory()
        model_val.fit(
            x,
            y,
            epochs=epochs_val,
            batch_size=batch_size_val,
            callbacks=[time_callback],
            verbose=False
        )

        batch_median = np.median(time_callback.batch_times[2:])
        # remove first batch to remove the effect of setup, and compensate with median batch time
        real_time_process_first_batchs.append(
            sum([batch_median] + time_callback.batch_times[1:]) * 1000
        )
        real_time_batchs.append(sum(time_callback.batch_times) * 1000)
        real_time_epochs.append(sum(time_callback.epoch_times) * 1000)
        real_time_start_ends.append(
            (time_callback.train_end_time - time_callback.train_start_time) * 1000
        )

        train_batch_numbers = math.ceil(data_size_val / batch_size_val) * epochs_val

        x_val = mtd_val.convert_model_data(
            model_val,
            50,
            m_config['optimizer'],
            m_config['loss'],
            batch_size_val,
            layer_na_fill=0,
            act_na_fill=0,
            scaler=scaler
        ).to_numpy()
        y_val_pred_batch = batch_model.predict(x_val)
        y_val_pred_batch = y_val_pred_batch.reshape(y_val_pred_batch.shape[0], )[0]
        y_val_preds_batch.append(y_val_pred_batch * train_batch_numbers)

        y_val_pred_setup = setup_model.predict(x_val)
        y_val_pred_setup = y_val_pred_setup.reshape(y_val_pred_setup.shape[0], )[0]
        y_val_preds_setup.append(y_val_pred_setup)

    # define a function to calculate error
    def cal_score(pred, real, absolute=False):
        pred = np.array(pred).copy()
        real = np.array(real).copy()
        if absolute:
            return abs((pred - real) / real)
        else:
            return (pred - real) / real

    # x-axis
    x = range(len(y_val_preds_batch))

    # only use prediction from batch model and see error for no setup time
    plt.scatter(x, cal_score(y_val_preds_batch, real_time_process_first_batchs))
    plt.plot(x, [0.15] * len(x), c='r', linewidth=10)
    plt.plot(x, [-0.15] * len(x), c='r', linewidth=10)
    plt.title('trucated batch time error')
    plt.show()

    # see error of setup time model
    plt.scatter(
        x,
        cal_score(
            y_val_preds_setup,
            np.array(real_time_batchs) - np.array(real_time_process_first_batchs)
        )
    )
    plt.plot(x, [0.15] * len(x), c='r', linewidth=10)
    plt.plot(x, [-0.15] * len(x), c='r', linewidth=10)
    plt.title('setup time error')
    plt.show()

    # see error for true model time prediction, combine results from batch model and setup model
    plt.scatter(
        x,
        cal_score(np.array(y_val_preds_setup) + np.array(y_val_preds_batch), real_time_start_ends)
    )
    plt.plot(x, [0.15] * len(x), c='r', linewidth=10)
    plt.plot(x, [-0.15] * len(x), c='r', linewidth=10)
    plt.title('real batch time error, added pred setup time')
    plt.show()
