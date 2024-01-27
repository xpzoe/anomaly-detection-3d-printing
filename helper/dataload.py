import tensorflow as tf
import tensorflow_io as tfio
import cv2 as cv
from os import listdir
from os.path import isdir
import numpy as np


def preprocess_my_image():
    path_input = input('enter path or drag and drop INPUT folder: ')
    path_input = path_input.strip('\"')
    path_input = path_input.strip('\'')
    path_input = path_input.replace('\\', '/')
    path_input += '/'
    if isdir(path_input):
        pass
    else:
        raise FileNotFoundError('no folder found')
    input_folder = path_input

    filelist = listdir(input_folder)
    image_set = []

    print("loading starts...")
    for file in filelist:
        cv_img = cv.imread(input_folder + file, cv.IMREAD_ANYDEPTH) / 65535. * 2 - 1

        image_set.append(cv_img)
    print("loading completed")

    image_set = tf.convert_to_tensor(tf.reshape(image_set, [-1, 128, 128, 1]))

    x_1 = np.zeros_like(image_set[:76])
    x_2 = np.zeros_like(image_set[76:])
    for i in range(74):
        x_1[i] = image_set[2*i]
        x_2[i] = image_set[2*i+1]
    x_1[75] = image_set[150]
    return image_set
