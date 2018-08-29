# Classification task for food, other, people.

We just take a pre-made network from top (which recognizes many different things), cut off several top layers, then putting own classifier there. And then quickly learn it on a small dataset [https://github.com/keras-team/keras/issues/4465](https://github.com/keras-team/keras/issues/4465) ImageNet itself has categories like food and person

First, we run the vgg model with a cut-off top on the data, and save the output to a file
We will get generated data bottleneck_features_train.npy and bottleneck_classes_train.npy (wich is done by vgg_train_save_outputs.py).
From these do the training "data" for our little model bottleneck_vgg_model.h5
Model learned with vgg_train_on_saved.py
Then lets try to evaluate it using vgg_evaluate_on_saved.py with small dataset

Useful links

*   keras-team [using pre trained VGG16 for another classification task](https://github.com/keras-team/keras/issues/4465)

*   Francois Chollet [Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)

*   Greg Chu [How to use transfer learning and fine-tuning in Keras and Tensorflow to build an image recognition system and classify (almost) any object](https://deeplearningsandbox.com/how-to-use-transfer-learning-and-fine-tuning-in-keras-and-tensorflow-to-build-an-image-recognition-94b0b02444f2)

Especially thanks to [hcl14](https://github.com/hcl14)
