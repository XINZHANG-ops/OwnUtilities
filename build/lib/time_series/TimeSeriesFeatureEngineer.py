import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from numpy import polyfit
from scipy.stats import boxcox
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.pyplot as plt


def create_average_feature(input_df, value_column, new_column_name, window, step):
    """
    This function will find the rolling window mean, but different from series rolling
    window mean, this function can also consider a step parameter

    since regular rolling mean can average values in the window, but every time it will only
    one step forward, this function solves that problem
    :param input_df:
    :param value_column:
    :param new_column_name:
    :param window:
    :param step:
    :return:
    """
    df = input_df.copy()
    df.loc[df.index[(np.arange(len(df)) + 1) % step == 0],
           new_column_name] = df[value_column].rolling(window=window).mean()
    return df


class TimeSeries:
    def __init__(self, inut_df, data_column, date_column, show_suggestion=False):
        if show_suggestion:
            print('If you need to resample your data, do that first')
        self.data = inut_df.copy()
        self.data_column = data_column
        self.date_column = date_column
        self.data[date_column] = pd.to_datetime(self.data[date_column])
        self.data.set_index(date_column, inplace=True)
        self.history = list()

    # feature engineering
    def lag_feature(self, lag=(1, 1), **kwargs):
        lag = tuple([lag[0], lag[1] + 1])
        series = self.data[self.data_column]
        for i in range(*lag):
            self.data['t-{}'.format(i)] = series.shift(i)
        if kwargs.get('inside_call', False):
            pass
        else:
            self.history.append([TimeSeries.lag_feature, [lag]])

    def binary_net_return_feature(self, gap=1, **kwargs):
        # gap simply for t to predict t+gap
        series = self.data[self.data_column]
        shifted1 = series.shift(gap - 1)
        shifted2 = series.shift(gap)
        first = np.array(list(shifted1))
        second = np.array(list(shifted2))
        net = list(first - second)
        binary_labels = []
        for net in net:
            if math.isnan(net):
                binary_labels.append(net)
            else:
                if net > 0:
                    binary_labels.append(1)
                else:
                    binary_labels.append(0)
        self.data['binary return of gap{}'.format(gap)] = binary_labels
        if kwargs.get('inside_call', False):
            pass
        else:
            self.history.append([TimeSeries.binary_net_return_feature, [gap]])

    def rolling_mean_feature(self, gap=1, past_days=10, **kwargs):
        series = self.data[self.data_column]
        shifted = series.shift(gap + 1)
        window = shifted.rolling(window=past_days)
        means = window.mean()
        self.data['mean of past{} gap{}'.format(past_days, gap)] = means
        if kwargs.get('inside_call', False):
            pass
        else:
            self.history.append([TimeSeries.rolling_mean_feature, [gap, past_days]])

    def rolling_variance_feature(self, gap=1, past_days=10, var=True, **kwargs):
        """
        :param past_days:
        :param var: False for std
        :return:
        """
        series = self.data[self.data_column]
        shifted = series.shift(gap + 1)
        window = shifted.rolling(window=past_days)
        if var:
            var_std = window.var()
            self.data['variance of past{} gap{}'.format(past_days, gap)] = var_std
        else:
            var_std = window.std()
            self.data['std of past{} gap{}'.format(past_days, gap)] = var_std
        if kwargs.get('inside_call', False):
            pass
        else:
            self.history.append([TimeSeries.rolling_variance_feature, [gap, past_days, var]])

    def rolling_max_feature(self, gap=1, past_days=10, **kwargs):
        series = self.data[self.data_column]
        shifted = series.shift(gap)
        window = shifted.rolling(window=past_days)
        value = window.max()
        self.data['max of past{} gap{}'.format(past_days, gap)] = value
        if kwargs.get('inside_call', False):
            pass
        else:
            self.history.append([TimeSeries.rolling_max_feature, [gap, past_days]])

    def rolling_min_feature(self, gap=1, past_days=10, **kwargs):
        series = self.data[self.data_column]
        shifted = series.shift(gap)
        window = shifted.rolling(window=past_days)
        value = window.min()
        self.data['min of past{} gap{}'.format(past_days, gap)] = value
        if kwargs.get('inside_call', False):
            pass
        else:
            self.history.append([TimeSeries.rolling_min_feature, [gap, past_days]])

    def expanding_feature(self, method=None, **kwargs):
        """
        :param method: 'min','max','mean','var','std'
        :param kwargs:
        :return:
        """
        series = self.data[self.data_column]
        window = series.expanding()
        if method == 'min':
            past = window.min()
            self.data['past_min'] = past
        elif method == 'max':
            past = window.max()
            self.data['past_max'] = past
        elif method == 'mean':
            past = window.mean()
            self.data['past_mean'] = past
        elif method == 'var':
            past = window.var()
            self.data['past_var'] = past
        elif method == 'std':
            past = window.std()
            self.data['past_std'] = past
        else:
            pass
        if kwargs.get('inside_call', False):
            pass
        else:
            self.history.append([TimeSeries.expanding_feature, [method]])

    def volatility_feature(self, gap=1, past_days=10, **kwargs):
        """
        read paper Predicting Stock Price Direction using Support Vector
        Machines page 6 for details

        :param past_days:
        :param kwargs:
        :return:
        """
        series = self.data[self.data_column]
        shifted1 = series.shift(gap)
        shifted2 = series.shift(gap + 1)
        first = np.array(list(shifted1))
        second = np.array(list(shifted2))
        vol = pd.Series((first - second) / second)
        volroll = vol.rolling(window=past_days)
        value = list(volroll.mean())
        self.data['volatility of past{} gap{}'.format(past_days, gap)] = value
        if kwargs.get('inside_call', False):
            pass
        else:
            self.history.append([TimeSeries.volatility_feature, [gap, past_days]])

    def momentum_feature(self, gap=1, past_days=10, **kwargs):
        """
        read paper Predicting Stock Price Direction using Support Vector
        Machines page 6 for details

        :param past_days:
        :param kwargs:
        :return:
        """
        series = self.data[self.data_column]
        shifted1 = series.shift(gap)
        shifted2 = series.shift(gap + 1)
        first = np.array(list(shifted1))
        second = np.array(list(shifted2))
        mom = list(first - second)
        momentum = []
        for net in mom:
            if math.isnan(net):
                momentum.append(net)
            else:
                if net > 0:
                    momentum.append(1)
                else:
                    momentum.append(-1)
        momentum_series = pd.Series(momentum)
        momentumroll = momentum_series.rolling(window=past_days)
        value = list(momentumroll.mean())
        self.data['momentum of past{} gap{}'.format(past_days, gap)] = value
        if kwargs.get('inside_call', False):
            pass
        else:
            self.history.append([TimeSeries.momentum_feature, [gap, past_days]])

    # data processing
    def resampling(self, sample_frame='D', interpolate_order=2, **kwargs):
        series = self.data[self.data_column]
        # Y for Year, D for Day, M for month, Q for quater, T for minute,A for year-end
        upsampled = series.resample('{}'.format(sample_frame)).mean()
        interpolated = upsampled.interpolate(method='spline', order=interpolate_order)
        self.data = pd.DataFrame(interpolated, columns=[self.data_column])

    def normalize_data(self, scaler_range=(0, 1), update_history=False):
        series = self.data[self.data_column]
        temp = pd.DataFrame(series)
        scaler = MinMaxScaler(feature_range=scaler_range)
        self.data[self.data_column] = scaler.fit_transform(temp)
        # use this to scaler.inverse_transform()
        self.scaler = scaler
        if update_history:
            for f, para in self.history:
                para = tuple(para)
                f(self, *para, inside_call=True)

    def detrend(self, update_history=False):
        series = self.data[self.data_column]
        X = [i for i in range(0, len(series))]
        X = np.reshape(X, (len(X), 1))
        y = series.values
        model = LinearRegression()
        model.fit(X, y)
        trend = model.predict(X)
        detrended = [y[i] - trend[i] for i in range(0, len(series))]
        self.data[self.data_column] = detrended
        self.trend = trend
        if update_history:
            for f, para in self.history:
                para = tuple(para)
                f(self, *para, inside_call=True)

    def deseasonality(self, seasonality=None, poly_degree=2, update_history=False):
        series = self.data[self.data_column]
        X = [i % (seasonality) for i in range(0, len(series))]
        y = series.values
        degree = poly_degree
        coef = polyfit(X, y, degree)
        curve = list()
        for i in range(len(X)):
            value = coef[-1]
            for d in range(degree):
                value += X[i]**(degree - d) * coef[d]
            curve.append(value)
        deseasonal = [y[i] - curve[i] for i in range(0, len(series))]
        self.data[self.data_column] = deseasonal
        self.seasonality = curve
        if update_history:
            for f, para in self.history:
                para = tuple(para)
                f(self, *para, inside_call=True)

    def box_cox_transform(self, update_history=False):
        self.data[self.data_column], lam = boxcox(self.data[self.data_column])
        self.lamb = lam
        if update_history:
            for f, para in self.history:
                para = tuple(para)
                f(self, *para, inside_call=True)

    def standardization(self, update_history=False):
        transformer = StandardScaler()
        series = np.array(self.data[self.data_column])
        series = series.reshape(series.shape[0], 1)
        transformer.fit(series)
        transformed = transformer.transform(series)
        transformed = list(transformed.reshape((transformed.shape[0], )))
        self.data[self.data_column] = transformed
        # use for inverse transform
        self.standardization_transformer = transformer
        if update_history:
            for f, para in self.history:
                para = tuple(para)
                f(self, *para, inside_call=True)

    def moving_average_smoothing(self, window=3, update_history=False, show_suggestion=False):
        if show_suggestion:
            print('Smoothing need your data is stationary')
        series = self.data[self.data_column]
        rolling = series.rolling(window=window)
        series_rolling_mean = rolling.mean()
        self.data[self.data_column] = series_rolling_mean.values
        if update_history:
            for f, para in self.history:
                para = tuple(para)
                f(self, *para, inside_call=True)

    # test and plots
    def adf_test(self, column=None):
        if column:
            series = self.data[column]
        else:
            series = self.data[self.data_column]
        X = series.values
        result = adfuller(X)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))

    def lag_plot(self, column=None, lag_list=None):
        """
        :param column:
        :param lag_list: receives lag_list if 9 ints
        :return:
        """
        if lag_list:
            if column:
                series = self.data[column]
            else:
                series = self.data[self.data_column]
            f = plt.figure(figsize=(15, 15))
            for index, lag in enumerate(lag_list):
                f.add_subplot(3, 3, index + 1)
                plt.title("lag {}".format(lag))
                lag_plot(series, lag=lag)
                plt.show()
        else:
            if column:
                series = self.data[column]
            else:
                series = self.data[self.data_column]
            lag_list = [int(i) for i in np.linspace(1, len(series), 9)]
            f = plt.figure(figsize=(15, 15))
            for index, lag in enumerate(lag_list):
                f.add_subplot(3, 3, index + 1)
                plt.title("lag {}".format(lag))
                lag_plot(series, lag=lag)
                plt.show()

    def line_plot(self, column=None):
        if column:
            series = self.data[column]
        else:
            column = self.data_column
            series = self.data[self.data_column]
        plt.title(column)
        series.plot(figsize=(15, 5))
        plt.show()

    def density_plot(self, column=None):
        if column:
            series = self.data[column]
        else:
            column = self.data_column
            series = self.data[self.data_column]
        plt.title(column)
        plt.hist(series)
        plt.show()

    def qq_plot(self, column=None):
        if column:
            series = self.data[column]
        else:
            column = self.data_column
            series = self.data[self.data_column]
        qqplot(series, line='r')
        plt.show()

    def acf_plot(self, column=None):
        if column:
            series = self.data[column]
        else:
            column = self.data_column
            series = self.data[self.data_column]
        plt.title(column)
        autocorrelation_plot(series)
        plt.show()

    def pacf_plot(self, lags=30, column=None):
        if column:
            series = self.data[column]
        else:
            column = self.data_column
            series = self.data[self.data_column]
        plot_pacf(series, lags=lags)
        plt.show()

    # get train and test
    def get_train_test_data(self, target_name, feture_names, test_size=.3):
        data = self.data.dropna()
        y = np.array(list(data[target_name]))
        features = tuple([list(data[name]) for name in feture_names])
        x = np.array(list(zip(*features)))
        # # LSTM receives 3 dim shape (#sample, #time step, #features(close price))
        x = x.reshape((x.shape[0], x.shape[1]))
        train_size = int(x.shape[0] * (1 - test_size))
        x_train, x_test, y_train, y_test = x[:train_size, :], x[train_size:, :], y[:train_size,
                                                                                   ], y[train_size:,
                                                                                        ]
        return x_train, x_test, y_train, y_test

    def get_k_train_1_test_data_for_boosting(
        self, target_name, feture_names, train_set_number=1, test_size=.3
    ):
        data = self.data.dropna()
        y = np.array(list(data[target_name]))
        features = tuple([list(data[name]) for name in feture_names])
        x = np.array(list(zip(*features)))
        # # LSTM receives 3 dim shape (#sample, #time step, #features(close price))
        x = x.reshape((x.shape[0], x.shape[1]))
        train_size = int(x.shape[0] * (1 - test_size))
        x_train, x_test, y_train, y_test = x[:train_size, :], x[train_size:, :], y[:train_size,
                                                                                   ], y[train_size:,
                                                                                        ]
        each_train_size = int(train_size / train_set_number)
        train_dict = dict()
        for i in range(train_set_number):
            if i + 1 == train_set_number:
                train_dict[i] = [x_train[i * each_train_size:, :], y_train[i * each_train_size:, ]]
            else:
                train_dict[i] = [
                    x_train[i * each_train_size:(i + 1) * each_train_size, :],
                    y_train[i * each_train_size:(i + 1) * each_train_size,
                            ]
                ]
        return train_dict, x_test, y_test
