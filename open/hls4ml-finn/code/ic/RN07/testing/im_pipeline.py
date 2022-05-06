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

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        print(filename)   # use this to create y_dtest!!!!
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return np.array(images)

while (SOME CONDITIONG):
    # read image somehow
    # img = cv2.imread(os.path.join(folder,filename))
    img = 32x32x3

    # if input is rectangular, find shortest side of rectangle image
    smallest_dim = np.min(np.array([img.shape[0], img.shape[1]]))

    # center crop relative to this image, so that we have square image
    img_sq = img[int(img.shape[0]/2)-int(smallest_dim/2):int(img.shape[0]/2)+int(smallest_dim/2),
                int(img.shape[1]/2)-int(smallest_dim/2):int(img.shape[1]/2)+int(smallest_dim/2),
                :]
    
    # interpolate to make image 32x32x3
    res = cv2.resize(img_sq, dsize=(32, 32), interpolation=cv2.INTER_NEAREST)

    # convert to input ready for neural net
    X_dtest = np.ascontiguousarray(res, dtype=np.float32)  # doesn't change shape, just turns every element to float instead of int
    X_dtest = X_dtest/256.

    # have model predict the image
    y_pred = model.predict(X_dtest)

    if y_pred == 0:
        print("Automobile")
    elif y_pred == 1:
        print("Airplane")
    elif y_pred == 2:
        print("Bird")
    elif y_pred == 3:
        print("Cat")
    elif y_pred == 4:
        print("Deer")
    elif y_pred == 5:
        print("Dog")
    elif y_pred == 6:
        print("Frog")
    elif y_pred == 7:
        print("Horse")
    elif y_pred == 8:
        print("Ship")
    elif y_pred == 9:
        print("Truck")
