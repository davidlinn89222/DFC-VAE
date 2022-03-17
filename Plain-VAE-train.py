# Importing the required packages and our VAE.py module
import os
from glob import glob
import numpy as np
from utils.VAE import VariationalAutoencoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# run params, this produces folders – run/vae/001_faces/images, viz, weights, like this
section = 'Plain-VAE'
run_id = '001'
data_name = 'faces'
RUN_FOLDER = 'run/{}/'.format(section)
RUN_FOLDER += '_'.join([run_id, data_name])

if not os.path.exists(RUN_FOLDER):
    os.makedirs(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

DATA_FOLDER = './data/'

INPUT_DIM = (128,128,3)
BATCH_SIZE = 32 # images for a batch
filenames = np.array(glob(os.path.join(DATA_FOLDER, '*/*.jpg')))
NUM_IMAGES = len(filenames)

# scaling the images and save into data_flow for training the network
data_gen = ImageDataGenerator(rescale=1./255)
data_flow = data_gen.flow_from_directory(
    DATA_FOLDER
    , target_size = INPUT_DIM[:2]
    , batch_size = BATCH_SIZE
    , shuffle = True
    , class_mode = 'input'
    , subset = "training"
)

vae = VariationalAutoencoder(
    input_dim = INPUT_DIM
    , z_dim=200
    , mode="PLAIN-VAE"
    , encoder_conv_filters=[32,64,64,64]
    , encoder_conv_kernel_size=[3,3,3,3]
    , encoder_conv_strides=[2,2,2,2]
    , decoder_conv_t_filters=[64,64,32,3]
    , decoder_conv_t_kernel_size=[3,3,3,3]
    , decoder_conv_t_strides=[2,2,2,2]
    , use_batch_norm=True
    , use_dropout=True
)

vae.save(RUN_FOLDER)
vae.encoder.summary() 
vae.decoder.summary() 

# parameters
LEARNING_RATE = 0.0005
R_LOSS_FACTOR = 10000 # This parameter required for tuning the network.
EPOCHS = 5  # 200 # here I am giving 1, but best results take 200 epochs
PRINT_EVERY_N_BATCHES = 100 # new image will produce from 100 batches of #input images
INITIAL_EPOCH = 0

vae._compile(LEARNING_RATE, R_LOSS_FACTOR)

vae.train_with_generator(
    data_flow
    , epochs = EPOCHS
    , steps_per_epoch = NUM_IMAGES / BATCH_SIZE
    , run_folder = RUN_FOLDER
    , print_every_n_batches = PRINT_EVERY_N_BATCHES
    , initial_epoch = INITIAL_EPOCH
)



