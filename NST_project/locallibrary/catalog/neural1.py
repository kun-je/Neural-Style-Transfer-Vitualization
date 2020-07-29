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
from numpy import array
import tensorflow as tf
import tensorflow.keras.preprocessing.image as img
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.optimizers import Adam


MODEL = VGG16()
CONTENT_LAYERS = ['block5_conv2']
STYLE_LAYERS = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']


def load_image(image_path):
    """
        Args:
            image_path (str): This takes a given an image path
        Returns:
            <class 'numpy.ndarray'> : This would convert the given image into array
            <class 'PIL.Image.Image'>: This would convert the given image into PIL format
    """
    IMG_SIZE = 224
    CHANNEL = 3
    image_array = img.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    image = img.img_to_array(image_array)
    image = image.reshape((1, IMG_SIZE, IMG_SIZE, CHANNEL))
    image = preprocess_input(image)
    return image, image_array


def plot_img(image_path):
    """
        This function shows a plotted graph of the given image
        Args:
            image_path (str): This would take an image path
    """
    image = load_image(image_path)[1]
    plt.imshow(image)
    plt.show()


def save_image():
    """TODO"""
    return 0

def MSE(matrix_content, matrix_generated):
    """
        Args:
            matrix_content (<class 'numpy.ndarray'>):
            matrix_generated (<class 'numpy.ndarray'>):
        Returns:
            int: A number made by perform substraction operation from each matrix (tensor), followed by
                squared operation with each substraction operation. The operation reduce mean is then applied.
    """
    return tf.reduce_mean(tf.square(matrix_content - matrix_generated))


def get_layer(image_path, layer_name):
    """
        Args:
            image_path (str): A given image path
            layer_name (str): A given layer name within the cnn model
        Returns:
            <class 'numpy.ndarray'> :

    """
    image = load_image(image_path)[0]
    layer = tf.keras.Model(inputs=MODEL.inputs, outputs=MODEL.get_layer(layer_name).output)
    return layer.predict(image)

def get_weights(layer_name):
    """
        Args:
            layer_name (str): A layer name within the cnn model

        Returns:
            List<int>: A lists of nump array containing weights corresponding to the given layer
    """
    for layer in MODEL.layers:
        if (layer_name == layer.name):
            return layer.get_weights()


def content_loss_function(c_image_path, g_image_path, layer_name):
    """
        Args:
            c_image_path (str): To take the content image path
            g_image_path (str): To take the generate image path

        Returns:
            int: The loss content. A low integer denotes the content is similar
            to the generated image. A high integer denotes the content is not similar
            to the generated image
    """
    WEIGHT = 0.5
    generated_layer = get_layer(g_image_path, layer_name)

    content_layer = get_layer(c_image_path, layer_name)
    loss = MSE(generated_layer, content_layer)
    return WEIGHT*loss


def gram_matrix(tensor):
    """
        Args:
            tensor (tensor): take 4D tensor

        Returns:
            gram (tensor) : gram matrix which is 2D array of the multiplication
            of the reshape matrix and its transpose
    """
    m_shape = []
    m_shape.append(tensor.shape[3])
    m_shape.append(tensor.shape[1]*tensor.shape[2])
    tensor = tf.reshape(tensor,m_shape)
    gram = tf.matmul(tensor,tf.transpose(tensor))
    return gram;

def style_loss_function(s_image_path, g_image_path, layer_name):
    """
        Args:
            c_image_path (str): To take the style image path
            g_image_path (str): To take the generate image path

        Returns:
            int: The loss content. A low integer denotes the content is similar
            to the generated image. A high integer denotes the content is not similar
            to the generated image

    """
    generated_layer = get_layer(g_image_path, layer_name)
    style_layer = get_layer(s_image_path, layer_name)

    #finding gram matrix of s and g image from perticular layer
    generated_gram = gram_matrix(generated_layer)
    style_gram = gram_matrix(style_layer)

    channel = 3
    img_size = 224 * 224
    
    #each layer of total style loss
    loss = MSE(generated_gram, style_gram)/(4*(channel**2)*(img_size**2))
    return loss

def total_variation_loss(g_image):
    weight = 30
    loss = weight*tf.reduce_sum(tf.image.total_variation(g_image))
    return loss

def style_grad_optimizer():
    x = tf.Variable(2, name = 'x', dtype = tf.float32)
    
    return 0

def optimizer(learning_rate, beta1, beta2):
    adam = tf.keras.optimizers.Adam(learning_rate,beta1,beta2)
    return adam


image_path = 'cat.jpeg'
image = load_image('cat.jpeg')
num = content_loss_function("cat.jpeg", "noise.jpg", CONTENT_LAYERS[0])
print(num)

s = style_loss_function("cat.jpeg", "noise.jpg", STYLE_LAYERS[0])
print(s)
