def get_RNN_model_features(keras_model):
    layers = [
        layer_info for layer_info in keras_model.get_config()['layers']
        if layer_info['class_name'] == 'SimpleRNN' or layer_info['class_name'] == 'LSTM'
        or layer_info['class_name'] == 'GRU' or layer_info['class_name'] == 'Dense'
    ]
    layer_sizes = [l['config']['units'] for l in layers]
    acts = [l['config']['activation'].lower() for l in layers]
    unroll = []
    for l in layers:
        try:
            unroll.append(l['config']['unroll'])
        except:
            unroll.append(None)
    layer_Type = [l['class_name'] for l in layers]
    return layer_sizes, acts, unroll, layer_Type
