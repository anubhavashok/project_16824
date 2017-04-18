import tensorflow as tf
from tensorflow.python.platform import flags
from cleverhans.attacks import fgsm
from cleverhans.utils import cnn_model
import keras
from keras import backend
from keras.utils import np_utils
from glob import glob
import numpy as np
import cv2


imgs = glob('./data/dogs/*.jpg')

sess = tf.InteractiveSession()
keras.backend.set_session(sess)
x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
raw_img = tf.placeholder(tf.float32, shape=(None, None, None, 3))
img = tf.image.resize_image_with_crop_or_pad(raw_img, 224, 224)
model = cnn_model(img_rows=224, img_cols=224, channels=3)
predictions = model(img)
adv_x = fgsm(img, predictions, eps=0.1)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

for i in imgs[:1]:
    data = cv2.imread(i)
    adv = sess.run(adv_x, {raw_img: [data], keras.backend.learning_phase(): 0})
    cv2.imwrite('adv.png', adv[0])
