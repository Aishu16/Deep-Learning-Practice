from keras.models import load_model
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils, to_categorical
from keras import backend as K

K.set_image_dim_ordering('th')
import numpy as np
import keras

X_train = sys.argv[1]
X_test = sys.argv[2]
X_train = np.load(X_train)
X_test = np.load(X_test)
Y_test = to_categorical(X_test, 10)

model = load_model(sys.argv[3])


def test():
    score, acc = model.evaluate(X_train, Y_test, batch_size=128)
    m = acc
    print("Accurac:", m)


test()
