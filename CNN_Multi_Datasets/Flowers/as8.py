import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from PIL import Image
from keras import applications
from keras.models import Sequential, Model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout, Input
from keras.models import Sequential
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras import backend
from keras.optimizers import Adam
from keras import optimizers
from keras.metrics import categorical_accuracy
import os
import sys
from keras.models import load_model

x = sys.argv[1]
script_dir = os.path.dirname(".")
training_set_path = os.path.join(script_dir, x)
test_set_path = os.path.join(script_dir, x)
input_size = (256, 256)
input_tensor = Input(shape=(256, 256, 3))
nb_train_samples = 3000
nb_validation_samples = 1000
batch_size = 16

train_datagen = ImageDataGenerator(  # rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(validation_split=0.33)

training_set = train_datagen.flow_from_directory(training_set_path,
                                                 target_size=input_size,
                                                 batch_size=batch_size,
                                                 subset="training",
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory(test_set_path,
                                            target_size=input_size,
                                            batch_size=batch_size,
                                            subset="validation",
                                            class_mode='categorical')

model = applications.ResNet50(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
# model = keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(224,224,3))

# Freeze the layers which you don't want to train. Here I am freezing the all layers.
for layer in model.layers[:]:
    layer.trainable = False

# Adding custom Layer
# We only add
x = model.output
x = Flatten()(x)
# Adding even more custom layers
x = Dropout(0.3)(x)
x = Dense(1000, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(500, activation="relu")(x)
predictions = Dense(5, activation="softmax")(x)

# creating the final model
model_final = Model(input=model.input, output=predictions)

# compile the model
# model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.01),metrics=["accuracy"])
model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                    metrics=["accuracy"])

# Train the model
model_final.fit_generator(
    training_set,
    samples_per_epoch=nb_train_samples,
    epochs=10,
    # steps_per_epoch = 8000,
    validation_data=test_set,
    nb_val_samples=nb_validation_samples
)

model_final.save(sys.argv[2] + '.h5')
