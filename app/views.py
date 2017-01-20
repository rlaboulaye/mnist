from app import app
from flask import request, render_template, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from skimage import io as skio
from skimage.color import rgb2gray
from skimage.transform import resize

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'mnist'))
import model

#Restore Convolutional MNIST Model
x = tf.placeholder("float", [None, 784])
sess = tf.Session()

with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2, variables = model.convolutional(x, keep_prob)
saver = tf.train.Saver(variables)
stored_model_location = os.path.dirname(__file__) + '/mnist/convolutional.ckpt'
saver.restore(sess, stored_model_location)

#Runs input through convolutional model
def convolutional(input):
    return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'bmp'])

def allowed_file(filename):
    return '.' in filename and \
           filename.lower().rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/process_photo', methods=['POST'])
def process_photo():
    request.get_data()
    file = request.files['digitPhoto']
    if file and allowed_file(file.filename):
        # get image from bmp file
        im = Image.open(file)
        pixels = np.array(im)
        print(pixels)
        # upload pixels to tensorflow model and output a digit
        return render_template('report.html') # how to pass tensorflow results to this template?
    return render_template('upload-error.html') # need to do generic error handling here

@app.route('/api/mnist', methods=['POST'])
def mnist():
    input = resize(rgb2gray(np.invert(skio.imread(request.files['image']))),(28,28)).reshape(1,784)
    output = convolutional(input)
    return jsonify(results=output)
