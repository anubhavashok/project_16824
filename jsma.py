import tensorflow as tf
from tensorflow.python.platform import flags
from cleverhans.attacks import jsma
from cleverhans.attacks_tf import jacobian_graph
from cleverhans.utils import cnn_model
import keras
from keras import backend
from keras.utils import np_utils
from keras.applications import vgg16
from glob import glob
import numpy as np
import cv2
import os


FLAGS = flags.FLAGS
FLAGS.nb_classes = 10

_gamma = 0.01
target = 2

imgs = glob('./data/dogs/*.jpg')

sess = tf.InteractiveSession()
keras.backend.set_session(sess)
raw_img = tf.placeholder(tf.float32, shape=(None, None, None, 3))
img = tf.image.resize_image_with_crop_or_pad(raw_img, 224, 224)
model = vgg16.VGG16(input_tensor=img, weights='imagenet')
predictions = model(img)
grads = jacobian_graph(predictions, img, 10) 
'''
adv_x, res, percent_perturb = jsma(sess, img, predictions, grads,
                                               image_tensor,
                                               target, theta=1, gamma=_gamma,
                                               increase=True, back='tf',
                                               clip_min=0, clip_max=1)
'''


for i in imgs:
    data = cv2.imread(i)
    data = sess.run(img, feed_dict={raw_img:[data]}) 
    adv_x, res, percent_perturb = jsma(sess, img, predictions, grads,
                                               data,
                                               target, theta=1, gamma=_gamma,
                                               increase=True, back='tf',
                                               clip_min=0, clip_max=1)
    #adversarial = sess.run(adv_x, feed_dict={raw_img: [data]})
    print(adv_x)
