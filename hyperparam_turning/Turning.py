from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
from functools import partial
from skopt import space, gp_minimize
import time


class call_back_bayesian:
    def __init__(self):
        self.time_stamps = []
        self.accuracies = []
        self.configs = []
        self.all = []

    def time_stamp_call(self, res):
        self.time_stamps.append(time.time())

    def accuracy_call(self, res):
        self.accuracies.append(res['func_vals'][-1])

    def config_call(self, res):
        self.configs.append(res['x_iters'][-1])

    def all_call(self, res):
        self.all.append(list(res.items()))


class HyperParamsTurning:
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

        if score_measure:
            self.score_sign = score_sign
            self.score_measure = score_measure
        else:
            self.score_measure = partial(f1_score, average='macro')
            self.score_sign = -1

    def bayesian_optimize(self, params, param_names, x, y, kfold_n_splits):
        params = dict(zip(param_names, params))
        model = self.model(**params)

        if self.x_test and self.y_test:
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

    def bayesian_train(self, n_calls=10, verbose=True):
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


def demo_bayesian():
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_digits
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier

    param_space = [(3, 15, 'max_depth'), (100, 600, 'n_estimators'),
                   (['gini', 'entropy'], 'criterion'), (0.01, 1, 'max_features')]

    accuracy_measurement = partial(f1_score, average='macro')

    X, y = load_digits(n_class=10, return_X_y=True)

    turning = HyperParamsTurning(
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

    result, bayesian_callback = turning.bayesian_train(n_calls=10, verbose=True)

    print('#################################')
    print('accuracy history:')
    print(bayesian_callback.accuracies)
    print('config history:')
    print(bayesian_callback.configs)
    print(f'Best result happened at {bayesian_callback.configs.index(result.x)+1}th trial')
    print('Best parameters are: ', dict(zip(turning.param_names, result.x)))
    print('Best score:', min(bayesian_callback.accuracies))

    plt.figure(figsize=(15, 8))
    plt.scatter(range(len(bayesian_callback.accuracies)), bayesian_callback.accuracies)
    plt.title('score changes by trials')
    plt.show()


demo_bayesian()
