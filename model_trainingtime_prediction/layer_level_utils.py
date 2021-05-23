import tensorflow as tf
import time
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense


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


def get_train_data(
    epochs=10,
    truncate_from=2,
    batch_size=32,
    act='relu',
    opt='SGD',
    verbose=False,
    dim_in=100,
    dim_out=100,
    loss='categorical_crossentropy'
):

    # build model
    model = Sequential()
    model.add(
        Dense(dim_out, input_dim=dim_in, activation=act, kernel_initializer=tf.ones_initializer())
    )
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    data_shape = model.get_config()['layers'][0]['config']['batch_input_shape'][1:]
    out_shape = model.get_config()['layers'][-1]['config']['units']
    x = np.ones((batch_size, *data_shape), dtype=np.float32)
    y = np.ones((batch_size, out_shape), dtype=np.float32)

    time_callback = TimeHistory()
    model.fit(
        x, y, epochs=epochs, batch_size=batch_size, callbacks=[time_callback], verbose=verbose
    )
    times_batch = np.array(time_callback.batch_times
                           )[truncate_from:] * 1000  # *1000 covert seconds to ms
    times_epoch = np.array(time_callback.epoch_times)[truncate_from:] * 1000

    return np.median(times_batch), np.median(times_epoch)
