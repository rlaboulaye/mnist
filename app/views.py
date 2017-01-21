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
    h_conv1, h_pool1, h_conv2, h_pool2, y2, variables = model.convolutional(x, keep_prob)
#    y2, variables = model.convolutional(x, keep_prob)
saver = tf.train.Saver(variables)
stored_model_location = os.path.dirname(__file__) + '/mnist/convolutional.ckpt'
saver.restore(sess, stored_model_location)

#Run input through convolutional model

def convolutional_layer1(input):
    return sess.run(h_conv1, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()

def convolutional_layer1_pooled(input):
    return sess.run(h_pool1, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()

def convolutional_layer2(input):
    return sess.run(h_conv2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()

def convolutional_layer2_pooled(input):
    return sess.run(h_pool2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()

def convolutional_prediction(input):
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

@app.route('/api/mnist', methods=['GET'])
def mnist():
    output = "API Calls\n"
    output += "Send bmp image file associated with 'image' key in POST request\n"
    output += "Apply up to convolution layer 1: /api/mnist/layer1\n"
    output += "Apply up to convolution layer 1 and pooling: /api/mnist/layer1pooled\n"
    output += "Apply up to convolution layer 2: /api/mnist/layer2\n"
    output += "Apply up to convolution layer 2 and pooling: /api/mnist/layer2pooled\n"
    output += "Apply entire model: /api/mnist/prediction"
    return output

@app.route('/api/mnist/layer1', methods=['POST'])
def layer1():
    input = resize(rgb2gray(np.invert(skio.imread(request.files['image']))),(28,28)).reshape(1,784)
    output = convolutional_layer1(input)
    return jsonify(results=output)

@app.route('/api/mnist/layer1pooled', methods=['POST'])
def layer1pooled():
    input = resize(rgb2gray(np.invert(skio.imread(request.files['image']))),(28,28)).reshape(1,784)
    output = convolutional_layer1_pooled(input)
    return jsonify(results=output)

@app.route('/api/mnist/layer2', methods=['POST'])
def layer2():
    input = resize(rgb2gray(np.invert(skio.imread(request.files['image']))),(28,28)).reshape(1,784)
    output = convolutional_layer2(input)
    return jsonify(results=output)

@app.route('/api/mnist/layer2pooled', methods=['POST'])
def layer2pooled():
    input = resize(rgb2gray(np.invert(skio.imread(request.files['image']))),(28,28)).reshape(1,784)
    output = convolutional_layer2_pooled(input)
    return jsonify(results=output)

@app.route('/api/mnist/prediction', methods=['POST'])
def prediction():
    input = resize(rgb2gray(np.invert(skio.imread(request.files['image']))),(28,28)).reshape(1,784)
    output = convolutional_prediction(input)
    return jsonify(results=output)
