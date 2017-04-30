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

from labels import labels

#imgs = glob('../dogs/*.jpg')
imgs = glob('./data/dogs_cropped/*.jpg')
sess = tf.InteractiveSession()
keras.backend.set_session(sess)
raw_img = tf.placeholder(tf.float32, shape=(None, None, 3))
#img = tf.image.resize_image_with_crop_or_pad(raw_img, 224, 224)
img = tf.image.resize_images(raw_img, [224, 224], align_corners=True)
model = vgg16.VGG16(input_tensor=tf.expand_dims(img, 0), weights='imagenet')
predictions = model(tf.expand_dims(img, 0)) 
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

for i in imgs:
    data = cv2.imread(i)
    #output = sess.run(img, {raw_img: data})
    #path = './data/dogs_cropped/' + os.path.basename(i)
    #cv2.imwrite(path, output[0])
    res = sess.run(predictions, {raw_img: data})
    #print(labels[np.argmax(res)])
    topk = res.argsort()[0][-10:]
    topk_correct = np.logical_and(topk >= 151, topk <= 275)
    if(sum(topk_correct) >= 1):
        print(labels[topk[topk_correct][0]])
    else:
        print('WRONG')
    #print([labels[j] for j in res.argsort()[0][-10:]])
