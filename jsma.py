import tensorflow as tf
from tensorflow.python.platform import flags
from cleverhans.attacks import jsma
from cleverhans.attacks_tf import jacobian_graph
from cleverhans.utils import cnn_model
import keras
from keras import backend
from keras.utils import np_utils
from keras.applications import vgg16
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from glob import glob
import numpy as np
import cv2
import os


FLAGS = flags.FLAGS
FLAGS.nb_classes = 10

_gamma = 0.1
target = 2

imgs = glob('./data/dogs/*.jpg')

sess = tf.InteractiveSession()
keras.backend.set_session(sess)
raw_img = tf.placeholder(tf.float32, shape=(None, None, None, 3))
img = tf.image.resize_image_with_crop_or_pad(raw_img, 224, 224)
#input = Input(shape=(None, 224, 224, 3), name='image_input')
vgg_16_conv = vgg16.VGG16(input_tensor=img, weights='imagenet', include_top=False)
x = Flatten(name='flatten')(vgg_16_conv.output)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(10, activation='softmax', name='predictions')(x)
model = Model(input=vgg_16_conv.input, output=x)
model.summary()
predictions = model(img)
grads = jacobian_graph(predictions, img, 10) 
sess.run(tf.global_variables_initializer())

print(sess.run(predictions, feed_dict={raw_img:[cv2.imread(imgs[0])]}))

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
