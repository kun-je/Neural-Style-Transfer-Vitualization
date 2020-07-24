#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 21:53:59 2020
@author: kun-je, runnily
NST project in spyder
Thisis project is done using "A Neural Algorithm of Artistic Style
 by. Leon A. Gatys,  Alexander S. Ecker, Matthias Bethge" as a reference
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import tensorflow.keras.preprocessing.image as img
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

MODEL = VGG16()
CONTENT_LAYERS = ['block5_conv2'] 

STYLE_LAYERS = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']


def load_image(image_path):
    #load style image and preprocess it for vgg-16 and return origianl image and
    #already preprocess image
    IMG_SIZE = 224
    CHANNEL = 3
    image = img.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    image = img.img_to_array(image)
    image = image.reshape((1, IMG_SIZE, IMG_SIZE, CHANNEL))
    image = preprocess_input(image)
    return image


def plot_img(load_image):
    #show context images
    plt.imshow(load_image)
    plt.show()
    
    
def save_image():
    #this is to save image in the databased
    return 0

def MSE(matrix_true, matrix_pred):
    #this loss function use mean square error to find
    #the different between the actual matrix and predicted matrix
    return tf.reduce_mean(tf.square(matrix_true - matrix_pred))


def get_layer(image_path, layer_name):
    #This would extract a certain layer
    image = load_image(image_path)
    layer = tf.keras.Model(inputs=MODEL.inputs, outputs=MODEL.get_layer(layer_name).output)
    return layer.predict(image)
   
def get_weights(layer_name):
    #This would get weights of a layer
    for layer in MODEL.layers:
        if (layer_name == layer.name):
            return layer.get_weights()


def content_loss_function(c_image_path, g_image_path, layer_name):
    #content loss function ensure that there not much different on activation function
    #on high layer between content and generated image
    WEIGHT = 0.5
    generated_layer = get_layer(g_image_path, layer_name)

    content_layer = get_layer(c_image_path, layer_name)
    loss = MSE(generated_layer, content_layer)
    return WEIGHT*loss


def gram_matrix(tensor):
    #gram matrix
    return 0

image_path = 'dog.jpg'
num = content_loss_function("dog.jpg", "noise.jpg", CONTENT_LAYERS[0])
print(num)