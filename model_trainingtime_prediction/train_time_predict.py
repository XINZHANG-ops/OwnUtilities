import os
import json
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

abs_path = os.path.dirname(os.path.abspath(__file__))


class training_pred_class:
    def __init__(self):
        self.data_points = 200
        self.smooth_window_size = 21
        self.smooth_polynomial_order = 3
        with open(os.path.join(abs_path, 'data', 'trainpredict.json')) as f:
            self.data = json.load(f)
        self.look_up_table = dict((int(k), v) for k, v in self.data['look_up_table'].items())

    @staticmethod
    def get_features_from_keras_model(keras_model_obj):
        layers = [i for i in keras_model_obj.get_config()['layers'] if i['class_name'] == 'Dense']
        return layers

    def predict(self, model, batch_size, optimizer, GPU_TYPE='P100'):
        data_points = self.data_points
        smooth_window_size = self.smooth_window_size
        smooth_polynomial_order = self.smooth_polynomial_order
        look_up_table = self.look_up_table

        filename = f'new_{optimizer}_batch_{batch_size}'

        data_dict = self.data[f'AggregateExp_NewModel_{GPU_TYPE}_{filename}']

        all_layer_batch = data_dict['all_layer_batch'][:data_points]
        layer_numbers = data_dict['layer_numbers'][:data_points]

        xdata = np.array(layer_numbers).astype(np.float)
        ydata = np.array([np.median(i) for i in all_layer_batch])

        y_smooth = savgol_filter(
            ydata, smooth_window_size, smooth_polynomial_order
        )  # window size 51, polynomial order 3
        f_interp1d = interp1d(xdata, y_smooth, kind='cubic')

        features = training_pred_class.get_features_from_keras_model(model)

        layer_batch_times = []
        for feature in features:
            Lbatch = look_up_table[batch_size][optimizer]
            layer_batch_times.append(Lbatch)

        sum_layers_batch = sum(layer_batch_times)
        multiplier = float(f_interp1d(len(features)))

        pred = sum_layers_batch / multiplier
        return pred


prediction_model = training_pred_class()
