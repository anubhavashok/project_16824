import os
import numpy as np
import cv2
import tensorflow as tf
import keras
from keras import backend
from keras.utils import np_utils
from keras.applications import vgg16
from flask import Flask, flash, jsonify, render_template, request, redirect, url_for 
from labels import labels
from imagenet_utils import preprocess_input

from flask import make_response
from functools import wraps, update_wrapper
from datetime import datetime

def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
        
    return update_wrapper(no_cache, view)


UPLOAD_FOLDER = '/Users/anubhav/Desktop/16824/project_16824/demo/static/'

sess = tf.InteractiveSession()
keras.backend.set_session(sess)
#raw_img = tf.placeholder(tf.float32, shape=(None, None, 3))
#img = tf.image.resize_images(raw_img, [224, 224], align_corners=True)
#model = vgg16.VGG16(input_tensor=tf.expand_dims(img, 0), weights='imagenet', include_top=False)
model = vgg16.VGG16(weights='imagenet', include_top=True)
#predictions = model.predict(tf.expand_dims(img, 0))
#print(predictions.shape)
#sess.run(tf.global_variables_initializer())
#sess.run(tf.local_variables_initializer())

def detect(im):
    return model.predict(np.expand_dims(im, 0))
    #return sess.run(predictions, feed_dict={raw_img: im})

# webapp
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route('/', methods=['GET', 'POST'])
@nocache
def main():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file:
            filename = 'dog.jpg'
            #filename = file.filename#secure_filename(file.filename)
            if os.path.exists('./static/dog.jpg'):
                os.remove('./static/dog.jpg')
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # Perform prediction here
            im = cv2.imread('./static/dog.jpg').astype(float)
            im = preprocess_input(np.expand_dims(im, 0))[0]
            im = cv2.resize(im, (224, 224))
            res = detect(im)
            topk = res.flatten().argsort()[-5:]
            print(res.flatten()[topk])
            print(topk)
            topk_correct = np.logical_and(topk >= 151, topk <= 275)
            if(sum(topk_correct) >= 1):
                dog_res = 'Dog! ' + labels[topk[topk_correct][0]]
            else:
                dog_res = 'Not dog :('
            return render_template('index.html', dog_res=dog_res)
            #return redirect(url_for('uploaded_file',
            #                        filename=filename))
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
