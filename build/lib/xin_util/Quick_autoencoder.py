"""
****************************************
 * @author: Xin Zhang
 * Date: 7/14/21
****************************************
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, GaussianNoise, Input
from tensorflow.keras.models import Model
from tensorflow.python.keras import regularizers
from tensorflow import keras
from tensorflow.keras import layers
from xin_util.Plotly_Plot import Categorical_Scatter


# AE
class Autoencoder(Model):
    def __init__(
        self,
        input_dim,
        first_layer_size=100,
        latent_dim=3,
        encoder_layer_num=4,
        drop_out=0.05,
        gaussian_noise=0.1
    ):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.first_layer_size = first_layer_size
        self.encoder_layer_num = encoder_layer_num
        self.input_dim = input_dim
        self.layer_sizes = [
            int(i) for i in np.linspace(first_layer_size, latent_dim, encoder_layer_num)
        ]

        encoder_layers = []
        for size in self.layer_sizes[:-1]:
            encoder_layers.append(
                Dense(size, activation='relu', activity_regularizer=regularizers.l1(1e-7))
            )
            encoder_layers.append(Dropout(drop_out))
            encoder_layers.append(GaussianNoise(gaussian_noise))
        encoder_layers.append(
            Dense(latent_dim, activation='relu', activity_regularizer=regularizers.l1(1e-7))
        )

        decoder_layers = []
        for size in self.layer_sizes[::-1][1:]:
            decoder_layers.append(Dense(size, activation='relu'))
        decoder_layers.append(Dense(input_dim, activation='sigmoid'))

        self.encoder = tf.keras.Sequential(encoder_layers)
        self.decoder = tf.keras.Sequential(decoder_layers)

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoders(
    X_train,
    epochs,
    batch_size,
    first_layer_size=100,
    latent_dim=3,
    encoder_layer_num=4,
    drop_out=0.05,
    gaussian_noise=0.1,
    optimizer='sgd'
):
    # Initialize and compile autoencoder model
    autoencoder = Autoencoder(
        X_train.shape[1],
        first_layer_size=first_layer_size,
        latent_dim=latent_dim,
        encoder_layer_num=encoder_layer_num,
        drop_out=drop_out,
        gaussian_noise=gaussian_noise
    )
    autoencoder.compile(optimizer=optimizer, loss='BinaryCrossentropy')
    # Fit data to autoencoder model
    autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, verbose=True)
    return autoencoder


def predict_autoencoders(X, model):
    # Make predictions on input data
    encodings = model.encoder.predict(X)
    return encodings


def set_seed_ae(
    seed_value,
    X_ae,
    labels,
    epochs,
    batch_size,
    first_layer_size=100,
    latent_dim=3,
    encoder_layer_num=4,
    drop_out=0.05,
    gaussian_noise=0.1,
    optimizer='sgd'
):
    from numpy.random import seed
    my_seed = seed_value
    seed(my_seed)
    import tensorflow
    tensorflow.random.set_seed(my_seed)

    autoencoder = train_autoencoders(
        X_ae,
        epochs,
        batch_size,
        first_layer_size=first_layer_size,
        latent_dim=latent_dim,
        encoder_layer_num=encoder_layer_num,
        drop_out=drop_out,
        gaussian_noise=gaussian_noise,
        optimizer=optimizer
    )
    encodings = predict_autoencoders(X_ae, autoencoder)
    cs = Categorical_Scatter(encodings, labels)
    cs.plot()


# VAE (this only take inout size 5 * 5 * 1 , need to ba modified)
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def build_variational_encoder_decoder(latent_dim=3):
    encoder_inputs = keras.Input(shape=(5, 5, 1))
    x = layers.Conv2D(32, 3, activation="relu", strides=1, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=1, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(latent_dim, ))
    x = layers.Dense(5 * 5 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((5, 5, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=1, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    return encoder, decoder


def set_seed_vae(seed_value, X_vae, labels, epochs, batch_size):
    from numpy.random import seed
    my_seed = seed_value
    seed(my_seed)
    import tensorflow
    tensorflow.random.set_seed(my_seed)

    encoder, decoder = build_variational_encoder_decoder()
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.SGD())
    vae.fit(X_vae, epochs=epochs, batch_size=batch_size)

    z_mean, _, _ = vae.encoder.predict(X_vae)
    cs = Categorical_Scatter(z_mean, labels)
    cs.plot()


def demo():
    def generate_spheres(dim=3, radiuses=(1, 2, 3), n_sample_each=300):
        """
        generate data on spheres (in respective dimension)

        @param dim: data space dimension
        @param radiuses: tuple, how many different spheres and their radius
        @param n_sample_each: how many samples for each category
        @return:
        """
        def sample_spherical(radius, npoints, ndim):
            vec = np.random.randn(ndim, npoints)
            vec /= np.linalg.norm(vec, axis=0)
            return vec * radius

        X_origin = np.array([])
        y = np.array([])
        for index, radius in enumerate(radiuses):
            if index == 0:
                X_origin = sample_spherical(radius, n_sample_each, dim).transpose()
            else:
                X_origin = np.concatenate(
                    (X_origin, sample_spherical(radius, n_sample_each, dim).transpose())
                )
            y = np.concatenate((y, np.ones(n_sample_each) * radius))
        return X_origin, y

    # generate 3d data and plot
    set_seed_ae(
        1,
        X_origin,
        y,
        3,
        64,
        first_layer_size=500,
        latent_dim=3,
        encoder_layer_num=5,
        optimizer='sgd'
    )
    set_seed_vae(1, X_origin.reshape((X_origin.shape[0], 5, 5, 1)), y, 1, 256)
