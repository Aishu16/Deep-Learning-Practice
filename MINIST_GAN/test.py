from keras.models import load_model
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils, to_categorical
from keras import backend as K
import matplotlib.pyplot as plt

K.set_image_dim_ordering('th')
import numpy as np
import keras
import sys
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Input
from keras.utils import np_utils, generic_utils, to_categorical
import os
from keras import backend as K

K.set_image_dim_ordering('th')

modelfile = sys.argv[1]

gen = load_model(modelfile)


def plot_generated_images(examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = gen.predict(noise)
    generated_images = generated_images.reshape(100, 28, 28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(sys.argv[2] + '.png')


plot_generated_images()
