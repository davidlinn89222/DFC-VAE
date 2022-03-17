import os
from glob import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
import utils.VAE as VAE
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

################ Hyper-Parameters and FLAGS #####################
BATCH_SIZE = 32
EPOCHS = 5
# MODE = "DFC-VAE-123" # Avaible options: PLAIN-VAE, DFC-VAE-345, DFC-VAE-123
MODE = "DFC-VAE-345"
# MODE = "PLAIN-VAE"
ALPHA = 1
BETA = 0.5
INITIAL_EPOCH = 0
LEARNING_RATE = 0.0005
PRINT_EVERY_N_BATCHES = 100
IMG_HEIGHT = 128
IMG_WIDTH = 128
Z_DIM = 200
section = "vae" # add
run_id = '001' # add
data_name = 'faces' # add
RUN_FOLDER = 'run/{}/'.format(section) # add
RUN_FOLDER += '_'.join([run_id, data_name]) # add
# RUN_FOLDER = os.path.join("./run", MODE)
DATA_FOLDER = "./data"
IMAGE_FOLDER = './data/celeb'
INPUT_DIM = (IMG_HEIGHT, IMG_WIDTH, 3) # Color-images only
INPUT_DIM = (128, 128, 3)
FILENAMES = np.array(glob(os.path.join(IMAGE_FOLDER, '*.jpg')))
NUM_IMAGES = len(FILENAMES)
R_LOSS_FACTOR = 10000

if not os.path.exists(RUN_FOLDER):
    os.makedirs(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))



##################################################################

# Print some useful information about images data
print("Dimension of Images: ", INPUT_DIM)
print("Total Images: ", NUM_IMAGES)
print("Data Folder Path: ", DATA_FOLDER)
print("Image Folder Path: ", IMAGE_FOLDER)
print("Run Folder Path: ", RUN_FOLDER)

# Generate Data Flow
data_gen = ImageDataGenerator(rescale = 1./255) 

# Data flow
data_flow = data_gen.flow_from_directory(
    DATA_FOLDER
    , target_size = INPUT_DIM[:2]
    , batch_size = BATCH_SIZE
    , shuffle = True
    , class_mode = 'input'
    , subset = "training"
)

# Build VAE model
vae = VAE.VariationalAutoencoder(
    input_dim = INPUT_DIM
    , alpha = ALPHA
    , beta = BETA
    , z_dim = Z_DIM
    , mode = MODE
    , encoder_conv_filters = [32,64,64,64]
    , encoder_conv_kernel_size = [3,3,3,3]
    , encoder_conv_strides = [2,2,2,2]
    , decoder_conv_t_filters = [64,64,32,3]
    , decoder_conv_t_kernel_size = [3,3,3,3]
    , decoder_conv_t_strides = [2,2,2,2]
    , use_batch_norm = True
    , use_dropout = True
    )

vae.save(RUN_FOLDER)

# Compile VAE model
vae._compile(LEARNING_RATE, 10000)

# Train VAE model
vae.train_with_generator(
    data_flow
    , epochs = EPOCHS
    , steps_per_epoch = int(NUM_IMAGES / BATCH_SIZE)
    , run_folder = RUN_FOLDER
    , print_every_n_batches = PRINT_EVERY_N_BATCHES
    , initial_epoch = INITIAL_EPOCH
)
