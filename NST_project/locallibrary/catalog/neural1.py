#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 21:53:59 2020
@author: kun-je
NST project in spyder
Thisis project is done using "A Neural Algorithm of Artistic Style
 by. Leon A. Gatys,  Alexander S. Ecker, Matthias Bethge" as a reference
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

model = VGG16()

#load style image and preprocess it for vgg-16 and return origianl image and
#already preprocess image
def load_image(file_name):
    #load image
    IMG_SIZE = 224
    CHANNEL = 3
    image = load_img(file_name, target_size=(IMG_SIZE, IMG_SIZE))
    #convert image to pixel in array
    img = img_to_array(image)
    img = img.reshape((-1, IMG_SIZE, IMG_SIZE, CHANNEL))
    #preprocess it for VGG-16,change the image to the tensor
    img = preprocess_input(img)   
    return image,img 



#show context image
def plot_img(load_image):
    plt.imshow(load_image)
    plt.show()
    
    
#this is to save image in the databased
def save_image():
    return 0


#this loss function use mean square error to find
#the different between the actual matrix and predicted matrix
def loss_function(matrix_true, matrix_pred):
    return tf.keras.losses.MSE(matrix_true,matrix_pred)


def gram_matrix():
    dic = {}
    for layer in model.layers:
        dic[layer.name] = layer.get_weights()
        print(layer.name, dic[layer])

def style_loss(tensor):
    #TODO
    pass



image = load_image('dog.jpg')
plot_img(image[0])