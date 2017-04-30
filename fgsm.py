import tensorflow as tf
from tensorflow.python.platform import flags
from cleverhans.attacks import fgsm
from cleverhans.utils import cnn_model
import keras
from keras import backend
from keras.utils import np_utils
from keras.applications import vgg16
from glob import glob
from imagenet_utils import preprocess_input
import numpy as np
import cv2
import os

imgs = glob('../dogs_box/*.jpg')
_eps = 0.1
sess = tf.InteractiveSession()
keras.backend.set_session(sess)
x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
raw_img = tf.placeholder(tf.float32, shape=(None, None, 3))
#img = tf.image.resize_image_with_crop_or_pad(raw_img, 224, 224)
img = tf.image.resize_images(raw_img, [224, 224], align_corners=True)
model = vgg16.VGG16(input_tensor=tf.expand_dims(img, 0), weights='imagenet')
predictions = model(tf.expand_dims(img, 0))
adv_x, signed_grad = fgsm(img, predictions, eps=_eps)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

for i in imgs:
    data = cv2.imread(i)
    data = preprocess_input(np.expand_dims(data, 0))[0]
    grad = sess.run(signed_grad, {raw_img: data, keras.backend.learning_phase(): 0})
    adv_path = './data/dogs_box_grad/'  + os.path.basename(i).split('.')[0] + '.jpg'
    print(adv_path)
    cv2.imwrite(adv_path, grad[0]+1)
