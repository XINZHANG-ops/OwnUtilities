"""
****************************************
 * @author: Xin Zhang
 * Date: 6/1/21
****************************************
"""
import time
import tensorflow.keras as keras
import pandas as pd
from tqdm import tqdm
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from random import sample
from sklearn.preprocessing import MinMaxScaler
import copy
import random
import collections

activation_fcts = [
    'relu', "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "exponential"
]
optimizers = ["sgd", "rmsprop", "adam", "adadelta", "adagrad", "adamax", "nadam", "ftrl"]
losses = ["mae", "mape", "mse", "msle", "poisson", "categorical_crossentropy"]
paddings = ["same", "valid"]


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


class ModelBuild:
    def __init__(
        self,
        DEFAULT_INPUT_SHAPE=(32, 32, 3),
        filter_lower=1,
        filter_upper=101,
        paddings=None,
        dense_lower=1,
        dense_upper=1001,
        activations=None,
        optimizers=None,
        losses=None
    ):
        self.kwargs_list: list
        self.layer_orders: list
        self.DEFAULT_INPUT_SHAPE = DEFAULT_INPUT_SHAPE
        self.activation_fcts = activation_fcts
        self.optimizers = optimizers
        self.losses = losses
        self.paddings = paddings

        OPTIONS = collections.defaultdict(dict)

        OPTIONS["Model"]["layer"] = [
            "Conv2D", "Dense", "MaxPooling2D", "Dropout", "Flatten"
        ]  # the model's layer can be either Conv2D or Dense
        OPTIONS["Compile"]["optimizer"
                           ] = optimizers if optimizers is not None else self.optimizers.copy()
        OPTIONS["Compile"]["loss"] = losses if losses is not None else self.losses.copy()
        OPTIONS["Dense"]["units"] = list(range(dense_lower, dense_upper))
        OPTIONS["Dense"]["activation"
                         ] = activations if activations is not None else self.activation_fcts.copy()
        OPTIONS["Conv2D"]["filters"] = list(range(filter_lower, filter_upper))
        OPTIONS["Conv2D"]["padding"] = paddings if paddings is not None else self.paddings.copy()
        OPTIONS["Conv2D"][
            "activation"] = activations if activations is not None else self.activation_fcts.copy()
        OPTIONS["MaxPooling2D"]["padding"
                                ] = paddings if paddings is not None else self.paddings.copy()
        OPTIONS["Dropout"]["rate"] = [0.1]

        self.options = OPTIONS

    def chooseRandomComb(self, options_layer, activations=None):
        res = dict()
        for k, v in options_layer.items():
            if k == "activation" and activations is not None:
                res[k] = random.choice(activations)
            else:
                res[k] = (random.sample(v, 1)[0])
        return res

    def generateRandomModelConfigList(self, layer_orders, input_shape=None):
        """
        Use global variable all_comb to generate random cnn model conf
        To build a model, pass the return to buildCnnModel method
        """
        if input_shape is None:
            input_shape = self.DEFAULT_INPUT_SHAPE

        def updateImageShape(_l, _kwargs, _image_shape):
            kernel_size: tuple
            if _l == "Conv2D":
                if type(_kwargs["kernel_size"]) == int:  # when kwargs["kernel_size"] was set by int
                    kernel_size = (_kwargs["kernel_size"], _kwargs["kernel_size"])
                else:
                    kernel_size = _kwargs["kernel_size"]
            elif _l == "MaxPooling2D":
                if type(_kwargs["pool_size"]) == int:  # when kwargs["kernel_size"] was set by int
                    # for program simplicity, I called pool_size as kernel_size
                    kernel_size = (_kwargs["pool_size"], _kwargs["pool_size"])
                else:
                    kernel_size = _kwargs["pool_size"]

            if type(_kwargs["strides"]) == int:  # when kwargs["strides"] was set by int
                strides = (_kwargs["strides"], _kwargs["strides"])
            else:
                strides = _kwargs["strides"]
            if _kwargs["padding"] == "valid":
                _image_shape[0] = (_image_shape[0] - kernel_size[0]) // strides[0] + 1
                _image_shape[1] = (_image_shape[1] - kernel_size[1]) // strides[1] + 1
            if _kwargs["padding"] == "same":
                if _image_shape[0] % strides[0] > 0:
                    _image_shape[0] = _image_shape[0] // strides[0] + 1
                else:
                    _image_shape[0] = _image_shape[0] // strides[0]
                if _image_shape[1] % strides[1] > 0:
                    _image_shape[1] = _image_shape[1] // strides[1] + 1
                else:
                    _image_shape[1] = _image_shape[1] // strides[1]
            assert _image_shape[0] > 0 and _image_shape[1] > 0
            return _image_shape

        def validKernelStridesSize(_l, _kwargs, _image_shape):
            if _l == "Conv2D":
                if type(_kwargs["kernel_size"]) == int:
                    kernel_size = (_kwargs["kernel_size"], _kwargs["kernel_size"])
                else:
                    kernel_size = _kwargs["kernel_size"]
            elif _l == "MaxPooling2D":
                if type(_kwargs["pool_size"]) == int:  # when kwargs["kernel_size"] was set by int
                    # for program simplicity, I called pool_size as kernel_size
                    kernel_size = (_kwargs["pool_size"], _kwargs["pool_size"])
                else:
                    kernel_size = _kwargs["pool_size"]

            if type(_kwargs["strides"]) == int:
                strides = (_kwargs["strides"], _kwargs["strides"])
            else:
                strides = _kwargs["strides"]
            judge = True
            if _l in ["Conv2D", "MaxPooling2D"]:
                judge = judge and (
                    kernel_size[0] <= _image_shape[0] and kernel_size[1] <= _image_shape[1]
                )
            judge = judge and (strides[0] <= _image_shape[0] and strides[1] <= _image_shape[1])
            if judge:
                return True
            else:
                return False

        options = self.options
        kwargs_list = []
        image_shape: list = list(input_shape[:2])
        image_shape_list: list = []
        # image_shape should end up in the same shape as model
        new_layer_orders = []
        max_strides = [3, 3]

        for i, lo in enumerate(layer_orders):
            if lo == "Dense":
                kwargs = self.chooseRandomComb(options["Dense"], options["Dense"]['activation'])
            elif lo == "Conv2D":
                if image_shape[0] == 1 or image_shape[1] == 1:
                    # if one of the image dim has only size one, we stop adding new conv2D
                    continue
                options_conv2d = options["Conv2D"].copy()
                # always ensure the kernel and strides size is smaller than the image
                options_conv2d["kernel_size"] = list(
                    zip(range(1, image_shape[0]), range(1, image_shape[1]))
                )

                options_conv2d["strides"] = [(1, 1)] * 10 + list(
                    zip(range(1, max_strides[0]), range(1, max_strides[1]))
                )
                kwargs = self.chooseRandomComb(options_conv2d)
                image_shape = updateImageShape(lo, kwargs, image_shape)
                max_strides = [
                    min(max_strides[0], max(1, image_shape[0])),
                    min(max_strides[1], max(1, image_shape[1]))
                ]
            elif lo == "MaxPooling2D":
                if image_shape[0] == 1 or image_shape[1] == 1:
                    # if one of the image dim has only size one, we stop adding new conv2D
                    continue
                options_maxpooling2d = options["MaxPooling2D"].copy()
                options_maxpooling2d["pool_size"] = list(
                    zip(range(1, image_shape[0]), range(1, image_shape[1]))
                )
                options_maxpooling2d["strides"] = [(1, 1)] * 10 + list(
                    zip(range(1, max_strides[0]), range(1, max_strides[1]))
                )
                kwargs = self.chooseRandomComb(options_maxpooling2d)
                image_shape = updateImageShape(lo, kwargs, image_shape)
                max_strides = [
                    min(max_strides[0], max(1, image_shape[0])),
                    min(max_strides[1], max(1, image_shape[1]))
                ]
            elif lo == "Dropout":
                kwargs = self.chooseRandomComb(options["Dropout"])
            elif lo == "Flatten":
                kwargs = {}
            # elif l == "AveragePooling2D":
            #   pass
            else:
                print("Error: layer order contained unsupported layer: %s" % lo)
            kwargs_list.append(kwargs)
            new_layer_orders.append(lo)
            image_shape_list.append(image_shape.copy())

        kwargs = {}
        for k in ["Compile", "Fit"]:
            kwargs[k] = {}
            for item in options[k].keys():
                kwargs[k][item] = random.sample(options[k][item], 1)[0]
        kwargs_list.append(kwargs)
        return kwargs_list, new_layer_orders, image_shape_list


class CnnRules:
    def __init__(
        self,
        conv_layer_num_lower=1,
        conv_layer_num_upper=10,
        max_pooling_prob=0.5,
        dense_layer_num_lower=1,
        dense_layer_num_upper=5
    ):
        self.conv_layer_num_lower = conv_layer_num_lower  # Rule: No Convolutional Layer After the First Dense Layer
        self.conv_layer_num_upper = conv_layer_num_upper
        self.max_pooling_prob = max_pooling_prob
        self.dense_layer_num_lower = dense_layer_num_lower
        self.dense_layer_num_upper = dense_layer_num_upper

    def gen_cnn_rule(self):
        conv_layer_num = np.random.randint(self.conv_layer_num_lower, self.conv_layer_num_upper)
        dense_layer_num = np.random.randint(self.dense_layer_num_lower, self.dense_layer_num_upper)

        rule_list = []
        for _ in range(conv_layer_num):
            rule_list.append('Conv2D')
            max_pooling_appear = np.random.choice([True, False],
                                                  size=1,
                                                  replace=True,
                                                  p=[
                                                      self.max_pooling_prob,
                                                      1 - self.max_pooling_prob
                                                  ])[0]
            if max_pooling_appear:
                rule_list.append('MaxPooling2D')

        rule_list.append('Flatten')

        rule_list.extend(['Dense'] * dense_layer_num)

        return rule_list


class gen_cnn2d:
    def __init__(
        self,
        input_shape_lower=8,
        input_shape_upper=256,
        conv_layer_num_lower=1,
        conv_layer_num_upper=50,
        filter_lower=1,
        filter_upper=101,
        dense_layer_num_lower=1,
        dense_layer_num_upper=5,
        dense_size_lower=1,
        dense_size_upper=1001,
        max_pooling_prob=.5,
        input_channels=None,
        paddings=None,
        activations=None,
        optimizers=None,
        losses=None
    ):
        self.input_shape_lower = input_shape_lower
        self.input_shape_upper = input_shape_upper
        self.input_channels = input_channels if input_channels is not None else [1, 3]
        self.conv_layer_num_lower = conv_layer_num_lower
        self.conv_layer_num_upper = conv_layer_num_upper
        self.filter_lower = filter_lower
        self.filter_upper = filter_upper
        self.dense_layer_num_lower = dense_layer_num_lower
        self.dense_layer_num_upper = dense_layer_num_upper
        self.dense_size_lower = dense_size_lower
        self.dense_size_upper = dense_size_upper
        self.max_pooling_prob = max_pooling_prob

        self.activations = [
            'relu', "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu",
            "exponential"
        ]
        self.optimizers = [
            "sgd", "rmsprop", "adam", "adadelta", "adagrad", "adamax", "nadam", "ftrl"
        ]
        self.losses = ["mae", "mape", "mse", "msle", "poisson", "categorical_crossentropy"]
        self.paddings = ["same", "valid"]

        self.activation_pick = activations if activations is not None else self.activations.copy()
        self.optimizer_pick = optimizers if optimizers is not None else self.optimizers.copy()
        self.loss_pick = losses if losses is not None else self.losses.copy()
        self.padding_pick = paddings if paddings is not None else self.paddings.copy()

    @staticmethod
    def nothing(x):
        return x

    def generate_cnn2d_model(self):
        cnn_rules = CnnRules(
            conv_layer_num_lower=self.conv_layer_num_lower,
            conv_layer_num_upper=self.conv_layer_num_upper,
            max_pooling_prob=self.max_pooling_prob,
            dense_layer_num_lower=self.dense_layer_num_lower,
            dense_layer_num_upper=self.dense_layer_num_upper
        )
        layer_orders = cnn_rules.gen_cnn_rule()
        input_shape = np.random.randint(self.input_shape_lower, self.input_shape_upper)
        input_channels = np.random.choice(self.input_channels, 1)[0]
        mb = ModelBuild(
            DEFAULT_INPUT_SHAPE=(input_shape, input_shape, input_channels),
            filter_lower=self.filter_lower,
            filter_upper=self.filter_upper,
            paddings=self.padding_pick,
            dense_lower=self.dense_size_lower,
            dense_upper=self.dense_size_upper,
            activations=self.activation_pick,
            optimizers=self.optimizer_pick,
            losses=self.loss_pick
        )
        kwargs_list, layer_orders, image_shape_list = mb.generateRandomModelConfigList(layer_orders)
        return kwargs_list, layer_orders, (input_shape, input_shape, input_channels)

    @staticmethod
    def build_cnn2d_model(kwargs_list, layer_orders):
        cnn2d = Sequential()
        for i, lo in enumerate(layer_orders):
            kwargs = kwargs_list[i]
            if lo == "Dense":
                cnn2d.add(Dense(**kwargs))
            elif lo == "Conv2D":
                cnn2d.add(Conv2D(**kwargs))
            elif lo == "MaxPooling2D":
                cnn2d.add(MaxPooling2D(**kwargs))
            elif lo == "Dropout":
                cnn2d.add(Dropout(**kwargs))
            elif lo == "Flatten":
                cnn2d.add(Flatten())
        kwargs = kwargs_list[-1]
        cnn2d.compile(metrics=['accuracy'], **kwargs["Compile"])
        return cnn2d

    def generate_model_configs(self, num_model_data=1000, progress=True):
        model_configs = []
        if progress:
            loop_fun = tqdm
        else:
            loop_fun = gen_cnn2d.nothing
        for i in loop_fun(range(num_model_data)):
            kwargs_list, layer_orders, input_shape = self.generate_cnn2d_model()
            model_configs.append([kwargs_list, layer_orders, input_shape])
        return model_configs


class cnn2d_model_train_data:
    def __init__(
        self, model_configs, batch_sizes=None, epochs=None, truncate_from=None, trials=None
    ):
        self.model_configs = []
        for info_list in model_configs:
            self.model_configs.append(info_list.copy())
        self.batch_sizes = batch_sizes if batch_sizes is not None else [2**i for i in range(1, 9)]
        self.epochs = epochs if epochs is not None else 10
        self.truncate_from = truncate_from if truncate_from is not None else 2
        self.trials = trials if trials is not None else 5
        self.activation_fcts = activation_fcts
        self.optimizers = optimizers
        self.losses = losses

    def get_train_data(self, progress=True):
        model_data = []
        model_configs = []
        if progress:
            loop_fun = tqdm
        else:
            loop_fun = gen_nn.nothing
        for info_list in self.model_configs:
            model_configs.append(info_list.copy())
        for model_config_list in loop_fun(model_configs):
            kwargs_list = model_config_list[0]
            layer_orders = model_config_list[1]
            input_shape = model_config_list[2]
            model = gen_cnn2d.build_cnn2d_model(kwargs_list, layer_orders)
            batch_size = sample(self.batch_sizes, 1)[0]
            batch_size_data_batch = []
            batch_size_data_epoch = []
            out_shape = model.get_config()['layers'][-1]['config']['units']
            x = np.ones((batch_size, *input_shape), dtype=np.float32)
            y = np.ones((batch_size, out_shape), dtype=np.float32)
            for _ in range(self.trials):
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

            model_config_list.append({
                'batch_size': batch_size,
                'batch_time': np.median(batch_times_truncated),
                'epoch_time': np.median(epoch_times_trancuted),
                'setup_time': np.sum(batch_size_data_batch) - sum(recovered_time),
                'input_dim': input_shape
            })
            model_data.append(model_config_list)
        return model_data

    def convert_config_data(
        self, model_data, max_layer_num=105, num_fill_na=0, name_fill_na=None, min_max_scaler=True
    ):

        feature_columns = [
            'layer_type', 'layer_size', 'kernel_size', 'strides', 'padding', 'activation',
            'optimizer', 'loss', 'batch_size', 'input_shape', 'channels'
        ]
        time_columns = ['batch_time', 'epoch_time', 'setup_time']
        feature_layer_types = ['Conv2D', 'MaxPooling2D', 'Dense']

        model_data_dfs = []
        time_rows = []
        for model_info in tqdm(model_data):
            data_rows = []
            kwargs_list = model_info[0]
            layer_orders = model_info[1]
            input_shape = model_info[2][0]
            channels = model_info[2][-1]
            train_times = model_info[3]
            for index, layer_type in enumerate(layer_orders):
                values = kwargs_list[index]
                if layer_type == 'Conv2D':
                    data_rows.append([
                        layer_type, values['filters'], values['kernel_size'][0],
                        values['strides'][0], values['padding'], values['activation'],
                        kwargs_list[-1]['Compile']['optimizer'], kwargs_list[-1]['Compile']['loss'],
                        train_times['batch_size'], input_shape, channels
                    ])
                elif layer_type == 'MaxPooling2D':
                    data_rows.append([
                        layer_type, num_fill_na, values['pool_size'][0], values['strides'][0],
                        values['padding'], name_fill_na, kwargs_list[-1]['Compile']['optimizer'],
                        kwargs_list[-1]['Compile']['loss'], train_times['batch_size'], input_shape,
                        channels
                    ])
                elif layer_type == 'Dense':
                    data_rows.append([
                        layer_type, values['units'], num_fill_na, num_fill_na, name_fill_na,
                        values['activation'], kwargs_list[-1]['Compile']['optimizer'],
                        kwargs_list[-1]['Compile']['loss'], train_times['batch_size'], input_shape,
                        channels
                    ])
                else:
                    pass
            time_rows.append([
                train_times['batch_time'], train_times['epoch_time'], train_times['setup_time']
            ])
            temp_df = pd.DataFrame(data_rows, columns=feature_columns)

            first_row = dict(temp_df.iloc[0])

            for opt in optimizers:
                first_row['optimizer'] = opt
                temp_df = temp_df.append(first_row, ignore_index=True)
            for lp in feature_layer_types:
                first_row['layer_type'] = lp
                temp_df = temp_df.append(first_row, ignore_index=True)
            for _pad in paddings:
                first_row['padding'] = _pad
                temp_df = temp_df.append(first_row, ignore_index=True)
            for _act in activation_fcts:
                first_row['activation'] = _act
                temp_df = temp_df.append(first_row, ignore_index=True)
            for _los in losses:
                first_row['loss'] = _los
                temp_df = temp_df.append(first_row, ignore_index=True)

            temp_df = pd.get_dummies(temp_df)
            temp_df = temp_df.drop(
                temp_df.index.tolist()
                [-len(optimizers + feature_layer_types + paddings + activation_fcts + losses):]
            )

            fill_empty_rows = dict((col_n, 0) for col_n in temp_df.columns)
            current_rows = temp_df.shape[0]
            compensate_count = max_layer_num - current_rows
            for _ in range(max([0, compensate_count])):
                temp_df = temp_df.append(fill_empty_rows, ignore_index=True)
            model_data_dfs.append(temp_df)
        time_df = pd.DataFrame(time_rows, columns=time_columns)
        if min_max_scaler:
            scaled_model_dfs = []
            scaler = MinMaxScaler()
            scaler.fit(pd.concat(model_data_dfs, axis=0).to_numpy())
            for data_df in model_data_dfs:
                scaled_data = scaler.transform(data_df.to_numpy())
                scaled_temp_df = pd.DataFrame(scaled_data, columns=temp_df.columns)
                scaled_model_dfs.append(scaled_temp_df)
            return scaled_model_dfs, time_df, scaler
        return model_data_dfs, time_df, None
