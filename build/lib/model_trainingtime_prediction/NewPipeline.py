"""
****************************************
 * @author: Xin Zhang
 * Date: 7/2/21
****************************************
"""

categorical_dictionary = {
    'activation': [
        'relu', "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "exponential"
    ],
    'optimizer': ["sgd", "rmsprop", "adam", "adadelta", "adagrad", "adamax", "nadam", "ftrl"],
    'loss': ["mae", "mape", "mse", "msle", "poisson", "categorical_crossentropy"],
    'use_bias': [True, False]
}
"""
'fillna_method' is for other layers without this property, what to fill in (we can put 0, None or whatever we need)
'use_name' is to change the original name if needed, since maybe we want to merge dense units and conv filters as one feature
'feature_status' is to use or not use this feature, with this we don't have to remove or delete items in this dictionary, just give a flag
"""


def value_grab(x):
    return x


def list_grab(x):
    return x[1]


def dict_grab(x):
    return x['class_name']


layer_feature_dictionary = {
    'Dense': [(
        'units', {
            'data_type': 'numerical',
            'fillna_method': None,
            'use_name': 'units',
            'feature_status': 1,
            'value_type': value_grab
        }
    ),
              (
                  'activation', {
                      'data_type': 'categorical',
                      'fillna_method': None,
                      'use_name': 'activation',
                      'feature_status': 0,
                      'value_type': value_grab
                  }
              ),
              (
                  'use_bias', {
                      'data_type': 'categorical',
                      'fillna_method': None,
                      'use_name': 'use_bias',
                      'feature_status': 1,
                      'value_type': value_grab
                  }
              )],
    'Activation': [(
        'activation', {
            'data_type': 'categorical',
            'fillna_method': None,
            'use_name': 'Activation',
            'feature_status': 1,
            'value_type': value_grab
        }
    )],
    'BatchNormalization': [(
        'momentum', {
            'data_type': 'numerical',
            'fillna_method': None,
            'use_name': 'momentum',
            'feature_status': 1,
            'value_type': value_grab
        }
    ),
                           (
                               'epsilon', {
                                   'data_type': 'numerical',
                                   'fillna_method': None,
                                   'use_name': 'epsilon',
                                   'feature_status': 1,
                                   'value_type': value_grab
                               }
                           )],
    'Conv2D': [(
        'filters', {
            'data_type': 'numerical',
            'fillna_method': None,
            'use_name': 'filters',
            'feature_status': 1,
            'value_type': value_grab
        }
    ),
               (
                   'batch_input_shape', {
                       'data_type': 'numerical',
                       'fillna_method': None,
                       'use_name': 'input_shape_size',
                       'feature_status': 1,
                       'value_type': lambda x: x[1]
                   }
               ),
               (
                   'batch_input_shape', {
                       'data_type': 'numerical',
                       'fillna_method': None,
                       'use_name': 'input_shape_channel',
                       'feature_status': 1,
                       'value_type': lambda x: x[3]
                   }
               ),
               (
                   'padding', {
                       'data_type': 'categorical',
                       'fillna_method': None,
                       'use_name': 'padding',
                       'feature_status': 1,
                       'value_type': value_grab
                   }
               ),
               (
                   'kernel_size', {
                       'data_type': 'numerical',
                       'fillna_method': None,
                       'use_name': 'kernel_size',
                       'feature_status': 1,
                       'value_type': lambda x: x[0]
                   }
               ),
               (
                   'strides', {
                       'data_type': 'numerical',
                       'fillna_method': None,
                       'use_name': 'strides',
                       'feature_status': 1,
                       'value_type': lambda x: x[0]
                   }
               ),
               (
                   'use_bias', {
                       'data_type': 'categorical',
                       'fillna_method': None,
                       'use_name': 'use_bias',
                       'feature_status': 1,
                       'value_type': value_grab
                   }
               )],
    'MaxPooling2D': [(
        'pool_size', {
            'data_type': 'numerical',
            'fillna_method': None,
            'use_name': 'pool_size',
            'feature_status': 1,
            'value_type': lambda x: x[0]
        }
    ),
                     (
                         'padding', {
                             'data_type': 'categorical',
                             'fillna_method': None,
                             'use_name': 'padding',
                             'feature_status': 1,
                             'value_type': value_grab
                         }
                     ),
                     (
                         'strides', {
                             'data_type': 'numerical',
                             'fillna_method': None,
                             'use_name': 'strides',
                             'feature_status': 1,
                             'value_type': lambda x: x[0]
                         }
                     )],
    'Dropout': [(
        'rate', {
            'data_type': 'numerical',
            'fillna_method': None,
            'use_name': 'rate',
            'feature_status': 1,
            'value_type': value_grab
        }
    )],
}


class TypeNoMatch(Exception):
    def __init__(self, m):
        self.message = m

    def __str__(self):
        return self.message


def check_layer_feature_dictionary_valid(layer_feature_dictionary):
    # we need to check same feature has same "data_type" and "fillna_method"
    from collections import defaultdict
    feature_data_type = defaultdict(set)
    feature_fillna_method = defaultdict(set)
    for class_n, conf_d in layer_feature_dictionary.items():
        for conf_type, instruct in conf_d:
            if instruct['feature_status']:
                feature_data_type[instruct['use_name']].add(instruct['data_type'])
                feature_fillna_method[instruct['use_name']].add(instruct['fillna_method'])
    for k in feature_data_type.keys():
        data_types = feature_data_type[k]
        fillna_method = feature_fillna_method[k]
        if len(data_types) > 1:
            raise TypeNoMatch(f'use_name "{k}" should have consistant data_type')
        if len(fillna_method) > 1:
            raise TypeNoMatch(f'use_name "{k}" should have consistant fillna_method')
    print('layer_feature_dictionary passed check')


check_layer_feature_dictionary_valid(layer_feature_dictionary)
