from keras.applications.vgg16 import VGG16

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input, decode_predictions

from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from keras.utils import np_utils

from keras.optimizers import SGD

from keras.models import Model
from keras.layers import Dense, Input, GlobalAveragePooling2D, Flatten, Dropout
from keras import backend as K


with open('bottleneck_features_train.npy', 'r') as f:
    train_data = np.load(f)
with open('bottleneck_classes_train.npy', 'r') as f:
    train_labels = np.load(f)
# shuffle
import random


def shuffle_list(a, b):
    l = list(zip(a, b))
    random.shuffle(l)
    return list(zip(*l))


train_data, train_labels = shuffle_list(train_data, train_labels)
train_data = list(train_data)
train_labels = list(train_labels)

# Spent a long time trying to adapt the data in the format it wants
#train_data = np.split(train_data, train_data[0].shape,axis=0)
#train_labels = np.split(train_labels, train_labels[0].shape,axis=0)
for i, val in enumerate(train_labels):
    train_labels[i] = val.ravel()

for i, val in enumerate(train_data):
    train_data[i] = val.ravel()


# we don't need it, already categorical
# change classes to categorical
#train_labels = np_utils.to_categorical(train_labels, 3)

# create just a two-layer model, which will accept the saved data from last vgg layer as input

inputs = Input(shape=train_data[0].shape)

#x = Flatten()(inputs)
'''
x = Dense(4096, activation='relu', kernel_initializer='glorot_normal')(inputs)
#x = Dropout(0.3)(x)
x = Dense(4096, activation='relu', kernel_initializer='glorot_normal')(x)
#x = Dropout(0.3)(x)
x = Dense(4096, activation='relu', kernel_initializer='glorot_normal')(x)
#x = Dropout(0.3)(x) # a little bit of droput to prevent agressive overfitting
x = Dense(1024, activation='relu', kernel_initializer='glorot_normal')(x)
predictions = Dense(3, activation='softmax', name='predictions')(x)
'''

x = Dense(4096, activation='relu', kernel_initializer='glorot_normal')(inputs)

#x = Dropout(0.3)(x)
x = Dense(4096, activation='relu', kernel_initializer='glorot_normal')(x)
#x = Dropout(0.3)(x)
x = Dense(4096, activation='relu', kernel_initializer='glorot_normal')(x)
# x = Dropout(0.3)(x) # a little bit of droput to prevent agressive overfitting
x = Dense(1024, activation='relu', kernel_initializer='glorot_normal')(x)
predictions = Dense(3, activation='softmax', name='predictions')(x)


model = Model(inputs=inputs, outputs=predictions)

# important ! It shuffles just batches as blocks, but not the elements. So if the data is ordered (as in out case), we will have the batches cosisting entirely of one class, so the model will stuck quickly
batch_size = 16

# You better repeat shuffle step between each epoch


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.2)
# binary_crossentropy = len(class_id_index) * categorical_crossentropy  - linearly dependent
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

model.compile(optimizer=sgd, loss='categorical_crossentropy',
              metrics=['accuracy'])

for i in range(2):
    model.fit(x=np.array(train_data), y=np.array(train_labels),
              epochs=1, batch_size=batch_size, shuffle=True)

    train_data, train_labels = shuffle_list(train_data, train_labels)
    train_data = list(train_data)
    train_labels = list(train_labels)

model.save('bottleneck_vgg_model.h5')
