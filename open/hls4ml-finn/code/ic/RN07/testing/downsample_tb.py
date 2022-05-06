import os
import tensorflow as tf
from tensorflow import keras
import qkeras
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Add
from tensorflow.keras.regularizers import l1_l2
from qkeras.qlayers import QDense, QActivation
from qkeras.qconvolutional import QConv2D
from qkeras.qconv2d_batchnorm import QConv2DBatchnorm
from qkeras.qpooling import QAveragePooling2D
from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import cv2


# UPLOAD MODEL
# path to model relative to downsample_tb.py
model_file_path = os.path.join("../training/trained_model", "model_best.h5")
co = {}
_add_supported_quantized_objects(co)
model = keras.models.load_model(model_file_path, custom_objects = co)

# print model summary
print(model.summary())


# UPLOAD TEST IMAGES
_, (X_test, y_test) = cifar10.load_data()
X_test = np.ascontiguousarray(X_test, dtype=np.float32)  # doesn't change shape, just turns every element to float instead of int
X_test = X_test/256.


num_classes = 10
y_test = tf.keras.utils.to_categorical(y_test, num_classes)   # one-hot encoding! ex. turn y_test of shape (2,) and 10 classes to shape (2, 10)

# METRICS FOR MODEL W/ TEST IMAGES IMGS
print("CIFAR10 Test Images Metrics: ")
# get predictions
y_pred = model.predict(X_test)

# evaluate with test dataset and share same prediction results
evaluation = model.evaluate(X_test, y_test)

auc = roc_auc_score(y_test, y_pred, average='weighted', multi_class='ovr')

print('Model test accuracy = %.3f' % evaluation[1])
print('Model test weighted average AUC = %.3f' % auc)

# I think this is the same as evaluation[1]
y_keras = y_pred
print("Keras Accuracy:  {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))))


print()
# load new test data of DOWNSAMPLED images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        print(filename)   # use this to create y_dtest!!!!
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return np.array(images)

print("Custom Test Images Uploaded: ")
X_dtest = load_images_from_folder("./downsampled_imgs")

X_dtest = np.ascontiguousarray(X_dtest, dtype=np.float32)  # doesn't change shape, just turns every element to float instead of int
X_dtest = X_dtest/256.

print("Custom Test Images Metrics: ")
# create the output array by arranging numbers 1-9, CORRESPONDING TO ORDER OF IMGS IN X_dtest
y_dtest = np.array([7, 3, 8, 4, 9, 2, 0, 1, 5, 6])
num_classes = 10
y_dtest = tf.keras.utils.to_categorical(y_dtest, num_classes)   # one-hot encoding! ex. turn y_test of shape (2,) and 10 classes to shape (2, 10)
print("Assigned classes: ", np.argmax(y_dtest, axis=1))
# METRICS FOR MODEL
# get predictions
y_pred = model.predict(X_dtest)

print("Predicted classes: ", np.argmax(y_pred, axis=1))

# evaluate with test dataset and share same prediction results
evaluation = model.evaluate(X_dtest, y_dtest)

auc = roc_auc_score(y_dtest, y_pred, average='weighted', multi_class='ovr')

print('Model test accuracy = %.3f' % evaluation[1])
print('Model test weighted average AUC = %.3f' % auc)

# I think this is the same as evaluation[1]
y_keras = y_pred
print("Keras Accuracy:  {}".format(accuracy_score(np.argmax(y_dtest, axis=1), np.argmax(y_keras, axis=1))))


