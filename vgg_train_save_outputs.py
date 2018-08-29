from keras.applications.vgg16 import VGG16

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input, decode_predictions

from keras.preprocessing.image import ImageDataGenerator

import numpy as np

from keras.models import Model
from keras.layers import Dense, Input, GlobalAveragePooling2D, Flatten
from keras import backend as K

import traceback


# Generate a model with all layers (with top)
vgg16 = VGG16(weights='imagenet', include_top=True)

# instaniate cut model
model = Model(inputs=vgg16.input, outputs=vgg16.layers[-3].output)


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['accuracy'])


# instantiate train generator
batch_size = 1  # to yield only one image at time

# fix https://github.com/keras-team/keras/issues/5475
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,

    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    shear_range=0.0,
    zoom_range=0.0,
    horizontal_flip=False
)


train_generator = train_datagen.flow_from_directory(
    'data/train',
    batch_size=batch_size,
    shuffle=False,  # we disable shuffling to make image order correspond to labels oder
    target_size=(224, 224)
)

# Does not work! Next network is not learning - perhaps the order is messed

# save classes
#class_mapping = train_generator.classes

# the predict_generator method returns the output of a model, given
# a generator that yields batches of numpy data
#bottleneck_features_train = model.predict_generator(train_generator, class_mapping.size,verbose=1)

class_mapping = []
bottleneck_features_train = []

i = 0

n_batches = train_generator.classes.size

for x, y in train_generator:

    i += 1

    if i % 100 == 0:
        print(i)

    if i > n_batches:
        break

    image = x[0]
    label = y[0]
    try:

        predict = model.predict(x)
    except:

        traceback.print_exc()

        # It has to be noted, that this does not work if shuffle is True (default). You will always get the filenames in the order they are first processed, not neccesarily in the order they are returned from the generator.

        print("please delete: " + train_generator.filenames[i])

        continue

    bottleneck_features_train.append(predict)
    class_mapping.append(label)


# Do not transform this data! Model.fit expexts a list of numpy arrays
#bottleneck_features_train = np.array(bottleneck_features_train)
#class_mapping = np.array(class_mapping)


# save the output as a Numpy array

with open('bottleneck_features_train.npy', 'w') as f:
    np.save(f, bottleneck_features_train)

with open('bottleneck_classes_train.npy', 'w') as f:
    np.save(f, class_mapping)


'''




#Add a layer where input is the output of the  second last layer 
x = Dense(1024, activation='relu')(vgg16.layers[-2].output)
x = Dense(1024, activation='relu')(x)
predictions = Dense(3, activation='softmax', name='predictions')(x)#(vgg16.layers[-2].output)






# this is the model we will train
model = Model(inputs=vgg16.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in vgg16.layers:
    layer.trainable = False
    
    
    
# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])



batch_size=16

train_datagen =  ImageDataGenerator(
    preprocessing_function=preprocess_input,
    
    rotation_range=30,
    width_shift_range=0.0,
    height_shift_range=0.0,
    shear_range=0.0,
    zoom_range=0.0,
    horizontal_flip=True
)


train_generator = train_datagen.flow_from_directory(
  'data/train',
  batch_size=batch_size,
  target_size=(224,224)
)


model.fit_generator(train_generator, steps_per_epoch=len(train_generator.filenames)//batch_size, epochs=1, workers=10,verbose=1)



model.save("vgg_model.h5")



img_path = 'food.jpg'

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print(preds)
#print('Predicted:', decode_predictions(preds)[0])





# evaluate



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
  'data/test',
  batch_size=batch_size,
  target_size=(224,224)
)


# this will take a while!
score = model.evaluate_generator(validation_generator, len(validation_generator.filenames) / batch_size, workers=8, use_multiprocessing = True)


print(score)
#[0.5243035554097443, 0.7842261904761905]

'''
