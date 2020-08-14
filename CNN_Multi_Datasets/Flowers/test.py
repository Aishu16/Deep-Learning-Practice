from keras.models import load_model
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils, to_categorical
# from keras import backend as K
# K.set_image_dim_ordering('th')
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

x = sys.argv[1]

input_size = (256, 256)
batch_size = 16
script_dir = os.path.dirname(".")
test_set_path = os.path.join(script_dir, x)

test_datagen = ImageDataGenerator(validation_split=0.33)

test_set = test_datagen.flow_from_directory(test_set_path,
                                            target_size=input_size,
                                            batch_size=batch_size,
                                            subset="validation",
                                            class_mode='categorical')

model = load_model(sys.argv[2])

Loss, score = model.evaluate_generator(test_set, workers=1)
print("Accuracy", score)
