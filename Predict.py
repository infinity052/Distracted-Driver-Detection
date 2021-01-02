import numpy as np
import os
import glob
import keras
import cv2
import random
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, \
                                       ZeroPadding2D
from datetime import datetime
from keras.utils import np_utils
from numpy.random import permutation
np.random.seed(2016)
use_cache = 1
color_type_global = 3
total_batch_size = 300


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

path = "./sample_imgs/*.jpg"

files = glob.glob(path)

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(64,64,3), kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu', kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=256, kernel_size=2, padding='same', activation='relu', kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=512, kernel_size=2, padding='same', activation='relu', kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(500, activation='relu', kernel_initializer='glorot_normal'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax', kernel_initializer='glorot_normal'))
model.load_weights(os.path.join('.','cache','model','distracted-09-1.00.hdf5'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

def get_im(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
    # mean_pixel = [103.939, 116.799, 123.68]
    # resized = resized.astype(np.float32, copy=False)

    # for c in range(3):
    #    resized[:, :, c] = resized[:, :, c] - mean_pixel[c]
    # resized = resized.transpose((2, 0, 1))
    # resized = np.expand_dims(img, axis=0)
    return resized
def getPrediction(num):
    prediction = "Prediction : "
    if num == 0:
        prediction+="Driver Focussed"
    elif num == 1 or num == 3:
        prediction+="Driver is Texting"
    elif num == 2 or num == 4:
        prediction += "Driver is on the phone"
    elif num == 5:
        prediction += "Driver is operating the radio"
    elif num == 6:
        prediction += "Driver is drinking"
    elif num == 7:
        prediction += "Driver is reaching behind"
    elif num == 8:
        prediction += "Driver is looking in the mirror"
    else:
        prediction += "Driver is talking to the passenger"
    return prediction
def load_img(img_rows, img_cols, color_type=1):
    X = []
    for file in files:
        img = get_im(file, img_rows, img_cols, color_type)
        X.append(img)
    return X
def normalize_imgs():
    img_rows = 64
    img_cols = 64
    X = load_img(img_rows, img_cols, color_type_global)
    X = np.array(X, dtype=np.uint8)
    X = X.astype('float32')
    mean_pixel = [103.-939, 116.779, 123.68]
    for c in range(3):
        X[:, :, :, c] = X[:, :, :, c] - mean_pixel[c]
    X /= 255
    return X
X = normalize_imgs()
print(X.shape)
predictions = model.predict_classes(X)
i = 0
for file in files:
    img = cv2.imread(file)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if int(predictions[i] == 0):
        color = (0, 255, 0)
    else:
        color = (0,0, 255)
    cv2.putText(img, getPrediction(int(predictions[i])), (10,450), font, 0.6, color, 2, cv2.LINE_AA)
    cv2.imshow(file,img)
    time = datetime.now()
    cv2.imwrite(os.path.join(".","Predictions", "img_"+str(i)+"_"+".jpg"), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    i += 1