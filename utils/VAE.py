import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, MaxPooling2D, UpSampling2D, LeakyReLU, Dropout, Layer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from utils.callbacks import CustomCallback, step_decay_schedule
from utils import padding

import numpy as np
import pickle

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# tf.config.experimental_run_functions_eagerly(True)
# tf.config.run_functions_eagerly(True)

# Whether to output all intermediates from functional control flow ops.
tf.compat.v1.experimental.output_all_intermediates(True)

# Build VariationalAutoencoder class
class VariationalAutoencoder():
    def __init__(self
        , input_dim
        , z_dim
        , mode
        , alpha=None
        , beta=None 
        , encoder_conv_filters=None
        , encoder_conv_kernel_size=None 
        , encoder_conv_strides=None
        , decoder_conv_t_filters=None
        , decoder_conv_t_kernel_size=None 
        , decoder_conv_t_strides=None 
        , use_batch_norm=None
        , use_dropout=None 
        ):

        self.name = 'variational_autoencoder'
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.alpha = alpha
        self.beta = beta
        self.mode = mode
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        # Setups for perceptual loss
        if self.mode == "DFC-VAE-123":
            print("Enter DFC-VAE-123 mode building.")
            self.selectedLayers = ["block1_conv1", "block2_conv1", "block3_conv1"]
            self.selectedLayer_weights = [1.0, 1.0, 1.0]
            self.build_loss_model()
            self._build_DFCVAE()

        elif self.mode == "DFC-VAE-345":
            print("Enter DFC-VAE-345 mode building.")
            self.selectedLayers = ["block3_conv1", "block4_conv1", "block5_conv1"]
            self.selectedLayer_weights = [1.0, 1.0, 1.0]
            self.build_loss_model()
            self._build_DFCVAE()

        elif self.mode == "PLAIN-VAE":
            print("Enter plain-VAE mode building.")
            self.n_layers_encoder = len(encoder_conv_filters)
            self.n_layers_decoder = len(decoder_conv_t_filters)
            self._build()
 

    def build_loss_model(self):

        self.vgg19 = VGG19(input_shape=self.input_dim, include_top=False, weights="imagenet")
        self.vgg19.trainable = False

        selectedOutputs = [self.vgg19.get_layer(l).output for l in self.selectedLayers]

        self.loss_model = Model(self.vgg19.input, selectedOutputs)


    def _build_DFCVAE(self):

        ### Encoder ###

        encoder_input = Input(shape=self.input_dim, name='encoder_input')
        x = encoder_input

        # """ conv1 """
        x = Conv2D(filters=32, kernel_size=(4, 4), strides=2, padding='same', name='encoder_conv_1')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # """ conv2 """
        x = Conv2D(filters=64, kernel_size=(4, 4), strides=2, padding='same', name='encoder_conv_2')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # """ conv3 """
        x = Conv2D(filters=128, kernel_size=(4, 4), strides=2, padding='same', name='encoder_conv_3')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # """ conv4 """
        x = Conv2D(filters=256, kernel_size=(4, 4), strides=2, padding='same', name='encoder_conv_4')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Flatten the samples
        shape_before_flattening = K.int_shape(x)[1:]

        x = Flatten()(x)

        # Fully-connected output layers
        self.mu = Dense(self.z_dim, name='mu')(x)
        self.log_var = Dense(self.z_dim, name='log_var')(x)
        self.encoder_mu_log_var = Model(encoder_input, (self.mu, self.log_var))

        def sampling(args):
            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            return mu + K.exp(log_var / 2) * epsilon

        encoder_output = Lambda(sampling, name='encoder_output')([self.mu, self.log_var])

        self.encoder = Model(encoder_input, encoder_output)

        ### Decoder ###
        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')

        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        x = Reshape(shape_before_flattening)(x)

        """ convTr1 """
        x = UpSampling2D(size=(2, 2), interpolation = "nearest", name="decoder_upsample_1")(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), padding = "valid", strides=1, name='decoder_conv_1')(x)
        x = padding.ReplicationPadding2D()(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        """ convTr2 """
        x = UpSampling2D(size=(2, 2), interpolation = "nearest", name="decoder_upsample_2")(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), padding = "valid", strides=1, name='decoder_conv_2')(x)
        x = padding.ReplicationPadding2D()(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        """ convTr3 """
        x = UpSampling2D(size=(2, 2), interpolation = "nearest", name="decoder_upsample_3")(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), padding = "valid", strides=1, name='decoder_conv_3')(x)
        x = padding.ReplicationPadding2D()(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        """ convTr4 """
        x = UpSampling2D(size=(2, 2), interpolation = "nearest", name="decoder_upsample_4")(x)
        x = Conv2D(filters=3, kernel_size=(3, 3), strides=1, padding='same', name='decoder_conv_4')(x)
        x = Activation('sigmoid')(x)

        decoder_output = x

        self.decoder = Model(decoder_input, decoder_output)

        ### THE FULL VAE
        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        self.model = Model(model_input, model_output)


    def _build(self):

        ### THE ENCODER
        encoder_input = Input(shape=self.input_dim, name='encoder_input')

        x = encoder_input

        for i in range(self.n_layers_encoder):
            conv_layer = Conv2D(
                filters = self.encoder_conv_filters[i]
                , kernel_size = self.encoder_conv_kernel_size[i]
                , strides = self.encoder_conv_strides[i]
                , padding = 'same'
                , name = 'encoder_conv_' + str(i)
                )

            x = conv_layer(x)

            if self.use_batch_norm:
                x = BatchNormalization()(x)

            x = LeakyReLU()(x)

            if self.use_dropout:
                x = Dropout(rate = 0.25)(x)

        shape_before_flattening = K.int_shape(x)[1:]

        x = Flatten()(x)

        self.mu = Dense(self.z_dim, name='mu')(x)
        self.log_var = Dense(self.z_dim, name='log_var')(x)
        
        self.encoder_mu_log_var = Model(encoder_input, (self.mu, self.log_var))

        def sampling(args):
            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            return mu + K.exp(log_var / 2) * epsilon

        encoder_output = Lambda(sampling, name='encoder_output')([self.mu, self.log_var])

        self.encoder = Model(encoder_input, encoder_output)

        ### THE DECODER
        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')

        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        x = Reshape(shape_before_flattening)(x)

        for i in range(self.n_layers_decoder):
            conv_t_layer = Conv2DTranspose(
                filters = self.decoder_conv_t_filters[i]
                , kernel_size = self.decoder_conv_t_kernel_size[i]
                , strides = self.decoder_conv_t_strides[i]
                , padding = 'same'
                , name = 'decoder_conv_t_' + str(i)
                )

            x = conv_t_layer(x)

            if i < self.n_layers_decoder - 1:
                if self.use_batch_norm:
                    x = BatchNormalization()(x)
                x = LeakyReLU()(x)
                if self.use_dropout:
                    x = Dropout(rate = 0.25)(x)
            else:
                x = Activation('sigmoid')(x)

        decoder_output = x

        self.decoder = Model(decoder_input, decoder_output)

        ### THE FULL VAE
        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        self.model = Model(model_input, model_output)

    def perceptual_loss(self, input_image, reconstruct_image):

        h1_list = self.loss_model(input_image)
        h2_list = self.loss_model(reconstruct_image)

        p_loss = 0.0
        for h1, h2 in zip(h1_list, h2_list):
            h1 = K.batch_flatten(h1)
            h2 = K.batch_flatten(h2)
            p_loss = p_loss + K.mean(K.square(h1 - h2), axis=-1)

        return p_loss


    def _compile(self, learning_rate, r_loss_factor):

        print("Penalty term for kl (alpha): ", self.alpha)
        print("Penalty term for loss (beta): ", self.beta)

        self.learning_rate = learning_rate

        def vae_r_loss(y_true, y_pred):
            r_loss = K.mean(K.square(y_true - y_pred), axis = [1,2,3])
            return r_loss*r_loss_factor

        def vae_ckl_loss(y_true, y_pred):
            kl_loss =  -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis = 1)
            return kl_loss*1

        # --------------------------------------------------------

        def vae_p_loss(y_true, y_pred):
            p_loss = self.perceptual_loss(y_true, y_pred)
            return p_loss*self.beta

        def vae_kl_loss(y_true, y_pred):
            kl_loss =  -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis = 1)
            return kl_loss*self.alpha


        ## Different definition for different mode

        if self.mode in ["DFC-VAE-123", "DFC-VAE-345"]:

            print("Compiling with DFC-VAE mode")

            def vae_dfc_loss(y_true, y_pred):
                p_loss = vae_p_loss(y_true, y_pred)
                kl_loss = vae_kl_loss(y_true, y_pred)
                return p_loss + kl_loss

        elif self.mode == "PLAIN-VAE":

            print("Compiling iwth Plain-VAE mode")

            def vae_loss(y_true, y_pred):
                r_loss = vae_r_loss(y_true, y_pred)
                kl_loss = vae_ckl_loss(y_true, y_pred)
                return r_loss + kl_loss

        optimizer = Adam(learning_rate=self.learning_rate)

        if self.mode in ["DFC-VAE-123", "DFC-VAE-345"]:
            self.model.compile(optimizer=optimizer, loss=vae_dfc_loss, metrics=[vae_p_loss, vae_kl_loss])

        elif self.mode == "PLAIN-VAE":
            self.model.compile(optimizer=optimizer, loss=vae_loss, metrics=[vae_r_loss, vae_ckl_loss])


    def save(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'viz'))
            os.makedirs(os.path.join(folder, 'weights'))
            os.makedirs(os.path.join(folder, 'images'))

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.input_dim
                , self.z_dim
                , self.mode
                , self.alpha
                , self.beta
                , self.encoder_conv_filters
                , self.encoder_conv_kernel_size
                , self.encoder_conv_strides
                , self.decoder_conv_t_filters
                , self.decoder_conv_t_kernel_size
                , self.decoder_conv_t_strides
                , self.use_batch_norm
                , self.use_dropout
                ], f)

        self.plot_model(folder)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches = 100, initial_epoch = 0, lr_decay = 1):

        custom_callback = CustomCallback(run_folder, print_every_n_batches, initial_epoch, self)
        lr_sched = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)
        checkpoint_filepath=os.path.join(run_folder, "weights/weights-{epoch:03d}-{loss:.2f}.h5")
        checkpoint1 = ModelCheckpoint(checkpoint_filepath, save_weights_only = True, verbose=1)
        checkpoint2 = ModelCheckpoint(os.path.join(run_folder, 'weights/weights.h5'), save_weights_only = True, verbose=1)

        callbacks_list = [checkpoint1, checkpoint2, custom_callback, lr_sched]

        self.model.fit(
            x_train
            , x_train
            , batch_size = batch_size
            , shuffle = True
            , epochs = epochs
            , initial_epoch = initial_epoch
            , callbacks = callbacks_list
        )


    def train_with_generator(self, data_flow, epochs, steps_per_epoch, run_folder, print_every_n_batches = 100, initial_epoch = 0, lr_decay = 1):

        # Create a CustomCallback object, which is defined in utils.callback.py
        custom_callback = CustomCallback(run_folder, print_every_n_batches, initial_epoch, self)

        # Call step_decay_schedule function, which is defined in utils.callback.py
        lr_sched = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)

        checkpoint_filepath=os.path.join(run_folder, "weights/weights-{epoch:03d}-{loss:.2f}.h5")
        checkpoint1 = ModelCheckpoint(checkpoint_filepath, save_weights_only = True, verbose=1, save_freq='epoch')
        checkpoint2 = ModelCheckpoint(os.path.join(run_folder, 'weights/weights.h5'), save_weights_only = True, verbose=1, save_freq='epoch')

        callbacks_list = [checkpoint1, checkpoint2, custom_callback, lr_sched]

        self.model.save_weights(os.path.join(run_folder, 'weights/weights.h5'))

        self.model.fit(
            data_flow
            , shuffle = True
            , epochs = epochs
            , initial_epoch = initial_epoch
            , callbacks = callbacks_list
            , steps_per_epoch = steps_per_epoch
            )

    def plot_model(self, run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder ,'viz/model.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.encoder, to_file=os.path.join(run_folder ,'viz/encoder.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.decoder, to_file=os.path.join(run_folder ,'viz/decoder.png'), show_shapes = True, show_layer_names = True)



