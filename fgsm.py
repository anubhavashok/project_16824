import tensorflow as tf
from tensorflow.python.platform import flags
from cleverhans.attacks import fgsm
from cleverhans.utils import cnn_model
import keras
from keras import backend
from keras.utils import np_utils
from glob import glob
import numpy as np


# Loads a single image from the dataset and computes jsma perturbation

# Path to the directory of the images
# NOTE: need to move images out of all folders into the main directory for this to work
dataset_dir = "./data/dogs/*.{}".format('jpg')

FLAGS = flags.FLAGS
FLAGS.nb_classes = 10
_eps = 0.1

# Read images in tensorflow symbolically
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once(dataset_dir))

image_reader = tf.WholeFileReader()

_, image_file = image_reader.read(filename_queue)

image = tf.image.decode_jpeg(image_file)
image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)


# Start a new session to show example output.
with tf.Session() as sess:
    keras.backend.set_session(sess)
    x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    model = cnn_model(img_rows=224, img_cols=224, channels=3)
    predictions = model(x)
    sess.run(tf.global_variables_initializer())
    
    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    image_tensor = sess.run([image])[0]
    print(image_tensor.shape)
    image_tensor = image_tensor[np.newaxis,:]
    
    # Target class
    adv_x = fgsm(x, predictions, eps=_eps)
    print(adv_x)
    
    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)


