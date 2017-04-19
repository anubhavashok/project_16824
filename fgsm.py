import tensorflow as tf
from tensorflow.python.platform import flags
from cleverhans.attacks import fgsm
from cleverhans.utils import cnn_model
import keras
from keras import backend
from keras.utils import np_utils
from keras.applications import vgg16
from glob import glob
import numpy as np
import cv2
import os

imgs = glob('./data/dogs/*.jpg')
_eps = 0.1
sess = tf.InteractiveSession()
keras.backend.set_session(sess)
x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
raw_img = tf.placeholder(tf.float32, shape=(None, None, None, 3))
img = tf.image.resize_image_with_crop_or_pad(raw_img, 224, 224)
#model = cnn_model(img_rows=224, img_cols=224, channels=3)
model = vgg16.VGG16(input_tensor= img, weights='imagenet')
#predictions = model(img)
#adv_x, signed_grad = fgsm(img, predictions, eps=_eps)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

for i in imgs:
    data = cv2.imread(i)
    #adv = sess.run(adv_x, {raw_img: [data], keras.backend.learning_phase(): 0})
    #grad = sess.run(signed_grad, {raw_img: [data], keras.backend.learning_phase(): 0})
    ri = sess.run(img, {raw_img: [data]})
    #adv_path = './data/dogs_grad_{}/'.format(_eps) + os.path.basename(i).split('.')[0] + '.jpg'
    path = './data/dogs_cropped/' + os.path.basename(i)
    #cv2.imwrite(adv_path, adv[0])
    cv2.imwrite(path, ri[0])
