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


def aspect_ratio(image_path):
    """
        Description:
            This get the image path of the input image, and uses the width and height to caculate the aspect ratio
        Input:
            image_path (str): The image path
        Returns:
                (int): The new resized width corresponding to the aspect ratio
                (int): The new resized height corresponding to the aspect ratio
    """
    image = img.load_img(image_path)
    image = img.img_to_array(image)
    width = image.shape[0]
    height = image.shape[1]
    new_width = 224
    new_height = (min(height,width)/max(width,height))*new_width
    if (new_height > 224 or new_height < 32):
        new_height = 224
    return int(new_width), int(new_height)

def load_image(image_path):
    """
        Description: 
            As we are using a pre-trained version VGG16 we have to resize and normalise
            the inputs.
        Args:
            image_path (str): This takes a given an image path
        Returns:
            <class 'numpy.ndarray'> : This would convert the given image into array
            <class 'PIL.Image.Image'>: This would convert the given image into PIL format
    """
    image_array = img.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    image = img.img_to_array(image_array)
    image = image.reshape((1, IMG_HEIGHT, IMG_WIDTH, CHANNEL))
    image = preprocess_input(image)
    return image, image_array

def deprocess_img(image):
    """
        Description: 
            This is used to reverse the depressing of the image. This is used in order
            to get the image.
        Args:
            image (<class 'numpy.ndarray'>) : Take in the given image in a preprocess format
        Returns:

    """
    temp_image = image
    temp_image = temp_image[0] # Gets one image, from samples
    temp_image = temp_image.reshape((IMG_HEIGHT, IMG_WIDTH, 3)) # converts it into 3-dimentions
    temp_image[:,:,0] += 103.939 #This adds the mean rgb back to the image, which the preprocess to off
    temp_image[:,:,1] += 116.779
    temp_image[:,:,2] += 123.68
    temp_image = temp_image[:,:,::-1]
    temp_image = np.clip(temp_image, 0, 255)
    return temp_image.astype('uint8')

def tensor_inputs(c_image_path, g_image_path, s_image_path):
    c_image = load_image(c_image_path)[0]
    g_image = load_image(g_image_path)[0]
    s_image = load_image(s_image_path)[0]
    return c_image, g_image, s_image
    

def plot_img(image_path):
    """
        Description: 
            This function shows a plotted graph of the given image
        Args:
            image_path (str): This would take an image path
    """
    image = load_image(image_path)[1]
    plt.imshow(image)
    plt.show()
    
    
def save_image(file_name, array_image):
    """
        Description: 
            This saves a given tensor image and saves the generated file into
            an output folder
        Args:
            file_name (string): This takes in the given file name
            array_image (): This takes in the given array
    """
    file_name = "output/{}".format(file_name)
    img.save_img(file_name, deprocess_img(array_image))
    return True

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


def get_layer(image, layer_name):
    """
        Args:
            image_path (str): A given image path
            layer_name (str): A given layer name within the cnn model
        Returns:
            <class 'numpy.ndarray'> :
    """
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


def content_loss_function(c_image, g_image, layer_name):
    #todo need to change doc string as type was changed
    """
        Args:
            c_image_path (tensor): To take the content image path
            g_image_path (str): To take the generate image path
            layer_name (str): To take in the layer name
        Returns:
            int: The loss content. A low integer denotes the content is similar 
            to the generated image. A high integer denotes the content is not similar
            to the generated image
    """
    WEIGHT = 0.5
    generated_layer = get_layer(g_image, layer_name)

    content_layer = get_layer(c_image, layer_name)
    loss = MSE(generated_layer, content_layer)
    return WEIGHT*loss

def gradient_content_loss(c_image, g_image, layer_name):
    pass

def gram_matrix(tensor):
    """
        Args: 
            tensor (tensor): take 3D tensor
            
        Returns:
            gram (tensor) : gram matrix which is 2D array of the multiplication 
            of the reshape matrix and its transpose
    """
    m_shape = []
    m_shape.append(tensor.shape[3])
    m_shape.append(tensor.shape[1]*tensor.shape[2])
    tensor = tf.reshape(tensor,m_shape)
    gram = tf.matmul(tensor,tf.transpose(tensor))
    return gram

def style_loss_function(s_image, g_image, layer_name):
    """
        Args:
            c_image_path (str): To take the style image path
            g_image_path (str): To take the generate image path
        Returns:
            int: The loss content. A low integer denotes the content is similar 
            to the generated image. A high integer denotes the content is not similar
            to the generated image
  
    """

    generated_layer = get_layer(g_image, layer_name)
    style_layer = get_layer(s_image, layer_name)
    
    #finding gram matrix of s and g image from perticular layer
    generated_gram = gram_matrix(generated_layer)
    style_gram = gram_matrix(style_layer)
    
    img_size = IMG_HEIGHT * IMG_WIDTH
    
    loss = MSE(generated_gram, style_gram)/(4*(CHANNEL**2)*(img_size**2))
    return loss
    
if __name__ == "__main__":
    MODEL = VGG16()
    CONTENT_LAYERS = ['block5_conv2'] 
    STYLE_LAYERS = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'] 
    image_path = "dog.jpg"
    noise_path = "noise.jpg"
    style_path = "style.jpg"
    IMG_WIDTH, IMG_HEIGHT = aspect_ratio(image_path) #optional if you want to apply an aspect path
    CHANNEL = 3

    c_image, g_image, s_image = tensor_inputs(image_path, image_path, style_path)

    gradient_content_loss(c_image, c_image, CONTENT_LAYERS[0])
    num = content_loss_function(c_image, c_image, CONTENT_LAYERS[0])
    print(num)


    save_image(image_path, c_image)
    s = style_loss_function(c_image, c_image, STYLE_LAYERS[0])
    print(s)