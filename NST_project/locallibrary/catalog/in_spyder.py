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
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

model = VGG16()

#load style image and preprocess it for vgg-16 and return origianl image and
#already preprocess image
def load_image(file_name):
    #load image
    image = load_img(file_name, target_size=(224, 224))
    #convert image to pixel in array
    img = img_to_array(image)
    img = img.reshape((-1, img.shape[0], img.shape[1], img.shape[2]))
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

#gram matrix
def gram_matrix(tensor):
    return 0

image = load_image('cat.jpeg')
plot_img(image[0])