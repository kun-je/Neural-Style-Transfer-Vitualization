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
import tensorflow as tf


from keras.applications.vgg16 import VGG16
model = VGG16()

#this is for read image from a file
def load_image():
    return 0

#this is to save image in the databased
def save_image():
    return 0

#this is for plotting iage using mathplotlib, to show image
def plot_image():
    return 0

#this loss function use mean square error to find
#the different between the actual matrix and predicted matrix
def loss_function(matrix_true, matrix_pred):
    return tf.keras.losses.MSE(matrix_true,matrix_pred)

#gram matrix
def gram_matrix(tensor):
    return 0