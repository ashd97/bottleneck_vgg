from keras.applications.vgg16 import VGG16

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input, decode_predictions

from keras.preprocessing.image import ImageDataGenerator

import numpy as np

from keras.models import Model
from keras.layers import Dense, Input, GlobalAveragePooling2D, Flatten, Dropout
from keras import backend as K

from keras.models import load_model


# Load a model with all layers (with top)
vgg16 = VGG16(weights='imagenet', include_top=True)

#x = Flatten()(inputs)
x = Dense(4096, activation='relu', kernel_initializer='glorot_normal')(
    vgg16.layers[-3].output)

#x = Dropout(0.3)(x)
x = Dense(4096, activation='relu', kernel_initializer='glorot_normal')(x)
#x = Dropout(0.3)(x)
x = Dense(4096, activation='relu', kernel_initializer='glorot_normal')(x)
# x = Dropout(0.3)(x) # a little bit of droput to prevent agressive overfitting
x = Dense(1024, activation='relu', kernel_initializer='glorot_normal')(x)
predictions = Dense(3, activation='softmax', name='predictions')(x)


model = Model(inputs=vgg16.input, outputs=predictions)


# load our model:

model2 = load_model("bottleneck_vgg_model.h5")


# transfer weights

# see model.summary() !

# watch the numbers! we omit dropouts
model.layers[-1].set_weights(model2.layers[-1].get_weights())
model.layers[-2].set_weights(model2.layers[-2].get_weights())
model.layers[-3].set_weights(model2.layers[-3].get_weights())
model.layers[-4].set_weights(model2.layers[-4].get_weights())
model.layers[-5].set_weights(model2.layers[-5].get_weights())


# evaluate:


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['accuracy'])


batch_size = 32

# test generator does not modify images
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    shear_range=0.0,
    zoom_range=0.0,
    horizontal_flip=False
)


validation_generator = test_datagen.flow_from_directory(
    'smalltest',
    batch_size=batch_size,
    target_size=(224, 224)
)


# this will take a while!
score = model.evaluate_generator(validation_generator, len(
    validation_generator.filenames) / batch_size, workers=1, use_multiprocessing=False)


print(score)
