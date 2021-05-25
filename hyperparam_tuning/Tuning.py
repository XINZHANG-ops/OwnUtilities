from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid, ParameterSampler
import numpy as np
from functools import partial
from skopt import space, gp_minimize
import time
import os
from tensorflow import keras
from kerastuner.tuners import BayesianOptimization as keras_BayesianOptimization
from kerastuner.tuners import RandomSearch as keras_Randomsearch
from kerastuner.tuners import Hyperband as keras_Hyperband
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout


class call_back_bayesian:
    def __init__(self):
        self.time_stamps = []
        self.accuracies = []
        self.configs = []
        self.all = []
        self.start_time = time.time()

    def time_stamp_call(self, res):
        self.time_stamps.append(time.time())

    def accuracy_call(self, res):
        self.accuracies.append(res['func_vals'][-1])

    def config_call(self, res):
        self.configs.append(res['x_iters'][-1])

    def all_call(self, res):
        self.all.append(list(res.items()))


class Bayesian:
    def __init__(
        self,
        model_callable,
        param_space,
        x_train,
        y_train,
        kfold_n_splits=5,
        score_sign=-1,
        score_measure=None,
        x_test=None,
        y_test=None
    ):
        """

        @param model_callable:
        @param param_space:
        @param x_train:
        @param y_train:
        @param n_calls:
        @param kfold_n_splits: this is used when no x_test, y_test given, cross validate score, but if x_test, y_test are given, not used
        @param score_sign: -1 if we want to max the value return by score_measure, 1 if we want to min it
        @param score_measure: default None for f1_score with avg is macro, callable for score calculation, take y_true as first arg, y_pred as second arg
        @param x_test: test data set data
        @param y_test: test data set label
        """
        self.model = model_callable
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.param_space = []
        self.param_names = []
        for param_config in param_space:
            self.param_names.append(param_config[-1])
            if isinstance(param_config[0], list):
                self.param_space.append(space.Categorical(param_config[0], name=param_config[-1]))
            elif isinstance(param_config[0], float):
                self.param_space.append(
                    space.Real(
                        low=param_config[0],
                        high=param_config[1],
                        prior='uniform',
                        name=param_config[-1]
                    )
                )
            elif isinstance(param_config[0], int):
                self.param_space.append(
                    space.Integer(low=param_config[0], high=param_config[1], name=param_config[-1])
                )
            else:
                raise
        self.kfold_n_splits = kfold_n_splits

        if score_measure is not None:
            self.score_sign = score_sign
            self.score_measure = score_measure
        else:
            self.score_measure = partial(f1_score, average='macro')
            self.score_sign = -1

    def bayesian_optimize(self, params, param_names, x, y, kfold_n_splits):
        params = dict(zip(param_names, params))
        model = self.model(**params)

        if self.x_test is not None and self.y_test is not None:
            model.fit(self.x_train, self.y_train)
            pred = model.predict(self.x_test)
            acc = self.score_measure(y_true=self.y_test, y_pred=pred)
            return self.score_sign * acc

        else:
            kf = StratifiedKFold(n_splits=kfold_n_splits)
            accuracies = []
            for idx in kf.split(X=x, y=y):
                train_idx, test_idx = idx[0], idx[1]
                xtrain = x[train_idx]
                ytrain = y[train_idx]

                xtest = x[test_idx]
                ytest = y[test_idx]

                model.fit(xtrain, ytrain)
                pred = model.predict(xtest)
                fold_acc = self.score_measure(y_true=ytest, y_pred=pred)
                accuracies.append(fold_acc)

            # note we multiply by -1 only when we want to max this score, if we deal with log, we want to remove -1
            return self.score_sign * np.mean(accuracies)

    def train(self, n_calls=10, verbose=True):
        optimization_function = partial(
            self.bayesian_optimize,
            param_names=self.param_names,
            x=self.x_train,
            y=self.y_train,
            kfold_n_splits=self.kfold_n_splits,
        )
        bayesian_callback = call_back_bayesian()
        result = gp_minimize(
            optimization_function,
            dimensions=self.param_space,
            n_calls=n_calls,
            n_initial_points=10,
            verbose=verbose,
            callback=[
                bayesian_callback.time_stamp_call, bayesian_callback.accuracy_call,
                bayesian_callback.config_call, bayesian_callback.all_call
            ]
        )
        return result, bayesian_callback


class GridSearch:
    def __init__(
        self,
        model_callable,
        param_space,
        x_train,
        y_train,
        kfold_n_splits=5,
        score_measure=None,
        x_test=None,
        y_test=None
    ):
        """

        @param model_callable:
        @param param_space:
        @param x_train:
        @param y_train:
        @param kfold_n_splits: this is used when no x_test, y_test given, cross validate score, but if x_test, y_test are given, not used
        @param score_sign:
        @param score_measure:
        @param x_test:
        @param y_test:
        """
        self.search_space = ParameterGrid(param_space)
        self.model = model_callable
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.kfold_n_splits = kfold_n_splits
        self.highest_score = 0
        self.lowest_score = 0

        if score_measure is not None:
            self.score_measure = score_measure
        else:
            self.score_measure = partial(f1_score, average='macro')

        self.history = []

    def train(self, verbose=True):
        for index, params in enumerate(self.search_space):
            if verbose:
                print(f'Step {index+1} starts, {len(self.search_space)-index-1} steps remaining')
            start_time = time.time()
            model = self.model(**params)

            if self.x_test is not None and self.y_test is not None:
                model.fit(self.x_train, self.y_train)
                pred = model.predict(self.x_test)
                acc = self.score_measure(y_true=self.y_test, y_pred=pred)
                # params, score, step_time
                self.history.append((params, acc, time.time() - start_time))

            else:
                kf = StratifiedKFold(n_splits=self.kfold_n_splits)
                accuracies = []
                for idx in kf.split(X=self.x_train, y=self.y_train):
                    train_idx, test_idx = idx[0], idx[1]
                    xtrain = self.x_train[train_idx]
                    ytrain = self.y_train[train_idx]

                    xtest = self.x_train[test_idx]
                    ytest = self.y_train[test_idx]

                    model.fit(xtrain, ytrain)
                    pred = model.predict(xtest)
                    fold_acc = self.score_measure(y_true=ytest, y_pred=pred)
                    accuracies.append(fold_acc)
                self.history.append((params, np.mean(accuracies), time.time() - start_time))
            if index == 0:
                self.highest_score = self.history[-1][1]
                self.lowest_score = self.history[-1][1]

            if self.history[-1][1] > self.highest_score:
                self.highest_score = self.history[-1][1]

            if self.history[-1][1] < self.lowest_score:
                self.lowest_score = self.history[-1][1]

            if verbose:
                print(
                    f'Step {index+1} ends, time spent: {round(self.history[-1][-1], 2)}s, step score: {round(self.history[-1][1], 2)}, highest: {round(self.highest_score, 2)}, loweest: {round(self.lowest_score, 2)}'
                )
                print('**********************')
        return sorted(self.history, key=lambda tup: tup[1], reverse=True)


class RandomSearch:
    def __init__(
        self,
        model_callable,
        param_space,
        x_train,
        y_train,
        kfold_n_splits=5,
        score_measure=None,
        x_test=None,
        y_test=None
    ):
        """

        @param model_callable:
        @param param_space: for int and categorical, a list of values, for real, use scipy.stats.distributions, an example is scipy.stats.distributions.uniform
        @param x_train:
        @param y_train:
        @param kfold_n_splits: this is used when no x_test, y_test given, cross validate score, but if x_test, y_test are given, not used
        @param score_sign:
        @param score_measure:
        @param x_test:
        @param y_test:
        """
        self.search_space = param_space
        self.model = model_callable
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.kfold_n_splits = kfold_n_splits
        self.highest_score = 0
        self.lowest_score = 0

        if score_measure is not None:
            self.score_measure = score_measure
        else:
            self.score_measure = partial(f1_score, average='macro')

        self.history = []

    def train(self, n_iter=10, random_state=None, verbose=True):
        rng = np.random.RandomState(random_state)
        search_space = list(ParameterSampler(self.search_space, n_iter=n_iter, random_state=rng))

        for index, params in enumerate(search_space):
            if verbose:
                print(f'Step {index + 1} starts, {len(search_space) - index - 1} steps remaining')
            start_time = time.time()
            model = self.model(**params)

            if self.x_test is not None and self.y_test is not None:
                model.fit(self.x_train, self.y_train)
                pred = model.predict(self.x_test)
                acc = self.score_measure(y_true=self.y_test, y_pred=pred)
                # params, score, step_time
                self.history.append((params, acc, time.time() - start_time))

            else:
                kf = StratifiedKFold(n_splits=self.kfold_n_splits)
                accuracies = []
                for idx in kf.split(X=self.x_train, y=self.y_train):
                    train_idx, test_idx = idx[0], idx[1]
                    xtrain = self.x_train[train_idx]
                    ytrain = self.y_train[train_idx]

                    xtest = self.x_train[test_idx]
                    ytest = self.y_train[test_idx]

                    model.fit(xtrain, ytrain)
                    pred = model.predict(xtest)
                    fold_acc = self.score_measure(y_true=ytest, y_pred=pred)
                    accuracies.append(fold_acc)
                self.history.append((params, np.mean(accuracies), time.time() - start_time))
            if index == 0:
                self.highest_score = self.history[-1][1]
                self.lowest_score = self.history[-1][1]

            if self.history[-1][1] > self.highest_score:
                self.highest_score = self.history[-1][1]

            if self.history[-1][1] < self.lowest_score:
                self.lowest_score = self.history[-1][1]

            if verbose:
                print(
                    f'Step {index + 1} ends, time spent: {round(self.history[-1][-1], 2)}s, step score: {round(self.history[-1][1], 2)}, highest: {round(self.highest_score, 2)}, loweest: {round(self.lowest_score, 2)}'
                )
                print('**********************')
        return sorted(self.history, key=lambda tup: tup[1], reverse=True)


class keras_dense_model_tune:
    def __init__(
        self,
        n_layers_min_max_step=(5, 10, 1),
        layer_size_min_max_step=(5, 100, 5),
        output_layer_size_act=(None, None),
        activations=None,
        optimizers=None,
        losses=None
    ):
        """

        @param output_layer_size_act: this one need to be separated because the output shape, but if set None, it will find the dim from y_train
        @param n_layers_min_max_step:
        @param layer_size_min_max_step:
        @param activations:
        @param optimizers:
        @param losses:
        """
        self.output_layer_size = output_layer_size_act[0]
        self.output_layer_act = output_layer_size_act[1]
        self.n_layers_min = n_layers_min_max_step[0]
        self.n_layers_max = n_layers_min_max_step[1]
        self.n_layers_step = n_layers_min_max_step[2]
        self.layer_size_min = layer_size_min_max_step[0]
        self.layer_size_max = layer_size_min_max_step[1]
        self.layer_size_step = layer_size_min_max_step[2]
        self.activations_default = [
            'relu', "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu"
        ]
        self.optimizers_default = [
            "sgd", "rmsprop", "adam", "adadelta", "adagrad", "adamax", "nadam", "ftrl"
        ]
        self.losses_default = ["categorical_crossentropy"]
        if activations is not None:
            self.activations = activations
        else:
            self.activations = self.activations_default.copy()
        if optimizers is not None:
            self.optimizers = optimizers
        else:
            self.optimizers = self.optimizers_default.copy()
        if losses is not None:
            self.losses = losses
        else:
            self.losses = self.losses_default.copy()

    def build_model(self, hp):
        model = keras.models.Sequential()
        for i in range(hp.Int('n_layers', min_value=self.n_layers_min, max_value=self.n_layers_max,
                              step=self.n_layers_step)):
            model.add(
                Dense(
                    units=hp.Int(
                        f'layer_{i}_size',
                        min_value=self.layer_size_min,
                        max_value=self.layer_size_max,
                        step=self.layer_size_step,
                    ),
                    activation=hp.Choice(f'layer_{i}_act', values=self.activations)
                )
            )
        model.add(Dense(units=self.output_layer_size, activation=self.output_layer_act))
        model.compile(
            optimizer=hp.Choice(f'optimizer', values=self.optimizers),
            loss=hp.Choice(f'loss', values=self.losses),
        )
        return model

    def ramdom_search_tune(
        self,
        x_train,
        y_train,
        x_test=None,
        y_test=None,
        epochs=10,
        batch_size=32,
        n_trials=10,
        executions_per_trial=1,
        save_dir=".",
        project_name='keras_model_tune',
        task_type='classification'
    ):
        """

        @param x_train:
        @param y_train:
        @param x_test:
        @param y_test:
        @param epochs:
        @param batch_size:
        @param n_trials:
        @param executions_per_trial:
        @param save_dir:
        @param project_name:
        @param task_type: string 'classification' or 'regression'
        @return:
        """

        if self.output_layer_size is None:
            self.output_layer_size = y_train.shape[1]
        if self.output_layer_act is None:
            self.output_layer_act = 'softmax'
        tuner = keras_Randomsearch(
            self.build_model,
            objective='val_accuracy' if task_type == 'classification' else 'val_loss',
            max_trials=n_trials,
            executions_per_trial=executions_per_trial,
            directory=save_dir,
            project_name=project_name
        )
        if x_test is None and y_test is None:
            x_test = x_train.copy()
            y_test = y_train.copy()
        tuner.search(
            x=x_train,
            y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test)
        )
        return tuner

    def bayesian_tune(
        self,
        x_train,
        y_train,
        x_test=None,
        y_test=None,
        epochs=10,
        batch_size=32,
        n_trials=10,
        executions_per_trial=1,
        save_dir=".",
        project_name='keras_model_tune',
        task_type='classification'
    ):
        if self.output_layer_size is None:
            self.output_layer_size = y_train.shape[1]
        if self.output_layer_act is None:
            self.output_layer_act = 'softmax'
        tuner = keras_BayesianOptimization(
            self.build_model,
            objective='val_accuracy' if task_type == 'classification' else 'val_loss',
            max_trials=n_trials,
            executions_per_trial=executions_per_trial,
            directory=save_dir,
            project_name=project_name
        )
        if x_test is None and y_test is None:
            x_test = x_train.copy()
            y_test = y_train.copy()
        tuner.search(
            x=x_train,
            y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test)
        )
        return tuner

    def hyperband_tune(
        self,
        x_train,
        y_train,
        x_test=None,
        y_test=None,
        epochs=10,
        max_epochs=15,
        factor=1,
        batch_size=32,
        executions_per_trial=1,
        save_dir=".",
        project_name='keras_model_tune',
        task_type='classification'
    ):
        if self.output_layer_size is None:
            self.output_layer_size = y_train.shape[1]
        if self.output_layer_act is None:
            self.output_layer_act = 'softmax'
        tuner = keras_Hyperband(
            self.build_model,
            objective='val_accuracy' if task_type == 'classification' else 'val_loss',
            max_epochs=max_epochs,
            factor=factor,
            executions_per_trial=executions_per_trial,
            directory=save_dir,
            project_name=project_name
        )
        if x_test is None and y_test is None:
            x_test = x_train.copy()
            y_test = y_train.copy()
        tuner.search(
            x=x_train,
            y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test)
        )
        return tuner


class keras_conv2d_model_tune:
    def __init__(
        self,
        n_conv_layers_min_max_step=(5, 10, 1),
        conv_layer_size_min_max_step=(32, 256, 32),
        kernel_size_min_max_step=(2, 3, 1),
        strides_min_max_step=(1, 2, 1),
        output_layer_size_act=(None, None),
        activations=None,
        optimizers=None,
        losses=None
    ):

        self.output_layer_size = output_layer_size_act[0]
        self.output_layer_act = output_layer_size_act[1]
        self.n_conv_layers_min = n_conv_layers_min_max_step[0]
        self.n_conv_layers_max = n_conv_layers_min_max_step[1]
        self.n_conv_layers_step = n_conv_layers_min_max_step[2]
        self.kernel_size_min = kernel_size_min_max_step[0]
        self.kernel_size_max = kernel_size_min_max_step[1]
        self.kernel_size_step = kernel_size_min_max_step[2]
        self.strides_min = strides_min_max_step[0]
        self.strides_max = strides_min_max_step[1]
        self.strides_step = strides_min_max_step[2]
        self.conv_layer_size_min = conv_layer_size_min_max_step[0]
        self.conv_layer_size_max = conv_layer_size_min_max_step[1]
        self.conv_layer_size_step = conv_layer_size_min_max_step[2]
        self.X_train = np.array([])
        self.activations_default = [
            'relu', "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu"
        ]
        self.optimizers_default = [
            "sgd", "rmsprop", "adam", "adadelta", "adagrad", "adamax", "nadam", "ftrl"
        ]
        self.losses_default = ["categorical_crossentropy"]
        if activations is not None:
            self.activations = activations
        else:
            self.activations = self.activations_default.copy()
        if optimizers is not None:
            self.optimizers = optimizers
        else:
            self.optimizers = self.optimizers_default.copy()
        if losses is not None:
            self.losses = losses
        else:
            self.losses = self.losses_default.copy()

    def build_model(self, hp):
        model = keras.models.Sequential()
        model.add(
            Conv2D(
                hp.Int(
                    'input_conv_units',
                    min_value=self.conv_layer_size_min,
                    max_value=self.conv_layer_size_max,
                    step=self.conv_layer_size_step,
                    default=32
                ),
                kernel_size=hp.Int(
                    'input_conv_kernel_size',
                    min_value=self.kernel_size_min,
                    max_value=self.n_conv_layers_max,
                    step=self.kernel_size_step,
                    default=3
                ),
                strides=hp.Int(
                    'input_conv_strides',
                    min_value=self.strides_min,
                    max_value=self.strides_max,
                    step=self.strides_step,
                    default=1
                ),
                input_shape=self.X_train.shape[1:],
                activation=hp.Choice(f'input_conv_act', values=self.activations)
            )
        )
        model.add(MaxPool2D(pool_size=(2, 2)))
        for i in range(hp.Int('n_conv_layers', min_value=self.n_conv_layers_min,
                              max_value=self.n_conv_layers_max, step=self.n_conv_layers_step,
                              default=2)):
            model.add(
                Conv2D(
                    hp.Int(
                        f'conv_{i}_units',
                        min_value=self.conv_layer_size_min,
                        max_value=self.conv_layer_size_max,
                        step=self.conv_layer_size_step,
                    ),
                    kernel_size=hp.Int(
                        f'conv_{i}_kernel_size',
                        min_value=self.kernel_size_min,
                        max_value=self.n_conv_layers_max,
                        step=self.kernel_size_step
                    ),
                    strides=hp.Int(
                        f'conv_{i}_strides',
                        min_value=self.strides_min,
                        max_value=self.strides_max,
                        step=self.strides_step
                    ),
                    activation=hp.Choice(f'conv_{i}_act', values=self.activations)
                )
            )
        model.add(Flatten())
        model.add(Dense(units=self.output_layer_size, activation=self.output_layer_act))
        model.compile(
            optimizer=hp.Choice(f'optimizer', values=self.optimizers),
            loss=hp.Choice(f'loss', values=self.losses),
        )
        return model

    def ramdom_search_tune(
        self,
        x_train,
        y_train,
        x_test=None,
        y_test=None,
        epochs=10,
        batch_size=32,
        n_trials=10,
        executions_per_trial=1,
        save_dir=".",
        project_name='keras_model_tune',
        task_type='classification'
    ):
        self.X_train = x_train.copy()
        if self.output_layer_size is None:
            self.output_layer_size = y_train.shape[1]
        if self.output_layer_act is None:
            self.output_layer_act = 'softmax'
        tuner = keras_Randomsearch(
            self.build_model,
            objective='val_accuracy' if task_type == 'classification' else 'val_loss',
            max_trials=n_trials,
            executions_per_trial=executions_per_trial,
            directory=save_dir,
            project_name=project_name
        )
        if x_test is None and y_test is None:
            x_test = x_train.copy()
            y_test = y_train.copy()
        tuner.search(
            x=x_train,
            y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test)
        )
        return tuner

    def bayesian_tune(
        self,
        x_train,
        y_train,
        x_test=None,
        y_test=None,
        epochs=10,
        batch_size=32,
        n_trials=10,
        executions_per_trial=1,
        save_dir=".",
        project_name='keras_model_tune',
        task_type='classification'
    ):
        self.X_train = x_train.copy()
        if self.output_layer_size is None:
            self.output_layer_size = y_train.shape[1]
        if self.output_layer_act is None:
            self.output_layer_act = 'softmax'
        tuner = keras_BayesianOptimization(
            self.build_model,
            objective='val_accuracy' if task_type == 'classification' else 'val_loss',
            max_trials=n_trials,
            executions_per_trial=executions_per_trial,
            directory=save_dir,
            project_name=project_name
        )
        if x_test is None and y_test is None:
            x_test = x_train.copy()
            y_test = y_train.copy()
        tuner.search(
            x=x_train,
            y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test)
        )
        return tuner

    def hyperband_tune(
        self,
        x_train,
        y_train,
        x_test=None,
        y_test=None,
        epochs=10,
        batch_size=32,
        n_trials=10,
        executions_per_trial=1,
        save_dir=".",
        project_name='keras_model_tune',
        task_type='classification'
    ):
        self.X_train = x_train.copy()
        if self.output_layer_size is None:
            self.output_layer_size = y_train.shape[1]
        if self.output_layer_act is None:
            self.output_layer_act = 'softmax'
        tuner = keras_Hyperband(
            self.build_model,
            objective='val_accuracy' if task_type == 'classification' else 'val_loss',
            max_trials=n_trials,
            executions_per_trial=executions_per_trial,
            directory=save_dir,
            project_name=project_name
        )
        if x_test is None and y_test is None:
            x_test = x_train.copy()
            y_test = y_train.copy()
        tuner.search(
            x=x_train,
            y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test)
        )
        return tuner


def demo_bayesian():
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_digits
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier

    param_space = [(3, 15, 'max_depth'), (100, 600, 'n_estimators'),
                   (['gini', 'entropy'], 'criterion'), (0.01, 1, 'max_features')]

    accuracy_measurement = partial(f1_score, average='macro')

    X, y = load_digits(n_class=10, return_X_y=True)

    tuning = Bayesian(
        model_callable=model,
        param_space=param_space,
        x_train=X,
        y_train=y,
        kfold_n_splits=5,
        score_sign=-1,
        score_measure=accuracy_measurement,
        x_test=None,
        y_test=None
    )

    result, bayesian_callback = tuning.train(n_calls=10, verbose=True)

    print('#################################')
    print('accuracy history:')
    print(bayesian_callback.accuracies)
    print('config history:')
    print(bayesian_callback.configs)
    print(f'Best result happened at {bayesian_callback.configs.index(result.x)+1}th trial')
    print('Best parameters are: ', dict(zip(tuning.param_names, result.x)))
    print('Best score:', min(bayesian_callback.accuracies))

    plt.figure(figsize=(15, 8))
    plt.scatter(range(len(bayesian_callback.accuracies)), bayesian_callback.accuracies)
    plt.title('score changes by trials')
    plt.show()


def demo_grid_search():
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_digits
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier

    param_space = {
        'max_depth': list(range(3, 16, 5)),
        'n_estimators': list(range(100, 601, 200)),
        'criterion': ['gini', 'entropy'],
        'max_features': np.linspace(0.01, 1, 3)
    }

    accuracy_measurement = partial(f1_score, average='macro')

    X, y = load_digits(n_class=10, return_X_y=True)

    tuning = GridSearch(
        model_callable=model,
        param_space=param_space,
        x_train=X,
        y_train=y,
        kfold_n_splits=5,
        score_measure=accuracy_measurement,
        x_test=None,
        y_test=None
    )

    result = tuning.train(verbose=True)

    print('#################################')
    print('accuracy history:')
    print([j for i, j, k in tuning.history])
    print('config history:')
    print([i for i, j, k in tuning.history])
    print('Best parameters are: ', result[0][0])
    print('Best score:', result[0][1])

    plt.figure(figsize=(15, 8))
    plt.scatter(range(len([j for i, j, k in tuning.history])), [j for i, j, k in tuning.history])
    plt.title('scores from all grids')
    plt.show()


def demo_random_search():
    import matplotlib.pyplot as plt
    from scipy.stats.distributions import uniform
    from sklearn.datasets import load_digits
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier

    param_space = {
        'max_depth': list(range(3, 16, 5)),
        'n_estimators': list(range(100, 601, 200)),
        'criterion': ['gini', 'entropy'],
        'max_features': uniform(0.01, 1)
    }

    accuracy_measurement = partial(f1_score, average='macro')

    X, y = load_digits(n_class=10, return_X_y=True)

    tuning = RandomSearch(
        model_callable=model,
        param_space=param_space,
        x_train=X,
        y_train=y,
        kfold_n_splits=5,
        score_measure=accuracy_measurement,
        x_test=None,
        y_test=None
    )

    result = tuning.train(n_iter=10, random_state=0, verbose=True)

    print('#################################')
    print('accuracy history:')
    print([j for i, j, k in tuning.history])
    print('config history:')
    print([i for i, j, k in tuning.history])
    print('Best parameters are: ', result[0][0])
    print('Best score:', result[0][1])

    plt.figure(figsize=(15, 8))
    plt.scatter(range(len([j for i, j, k in tuning.history])), [j for i, j, k in tuning.history])
    plt.title('scores from all random params')
    plt.show()


def keras_dense_turning_demo():
    import os
    from tensorflow import keras
    from keras.datasets import mnist

    kdm = keras_dense_model_tune(
        n_layers_min_max_step=(5, 10, 1),
        layer_size_min_max_step=(50, 100, 5),
        output_layer_size_act=(None, 'softmax'),
        activations=['relu'],
        optimizers=['SGD'],
        losses=['categorical_crossentropy']
    )
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    image_vector_size = 28 * 28
    x_train = x_train.reshape(x_train.shape[0], image_vector_size)
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)

    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    tuner = kdm.bayesian_tune(
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=10,
        batch_size=32,
        n_trials=10,
        executions_per_trial=1,
        save_dir=os.getcwd(),
        project_name='keras_dense_turning_demo'
    )
    tuner.results_summary()


def keras_conv_turning_demo():
    import os
    from tensorflow import keras
    from keras.datasets import fashion_mnist

    kdm = keras_conv2d_model_tune(
        n_conv_layers_min_max_step=(5, 10, 1),
        conv_layer_size_min_max_step=(32, 256, 32),
        kernel_size_min_max_step=(3, 3, 1),
        strides_min_max_step=(1, 1, 1),
        output_layer_size_act=(None, 'softmax'),
        activations=['relu'],
        optimizers=['SGD'],
        losses=['categorical_crossentropy']
    )
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    tuner = kdm.bayesian_tune(
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=10,
        batch_size=32,
        n_trials=2,
        executions_per_trial=1,
        save_dir=os.getcwd(),
        project_name='keras_conv_turning_demo'
    )
    tuner.results_summary()
