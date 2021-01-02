import numpy as np

import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, \
                                       ZeroPadding2D

# from keras.layers.normalization import BatchNormalization
# from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
# from sklearn.metrics import log_loss
from numpy.random import permutation

num_batches = 20
np.random.seed(2016)
use_cache = 1
# color type: 1 - grey, 3 - rgb
color_type_global = 3

# color_type = 1 - gray
# color_type = 3 - RGB
total_batch_size = 512


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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


def get_driver_data():
    dr2 = dict()
    path = os.path.join('.','dataset' , 'driver_imgs_list.csv')
    print('Read drivers data')
    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr2[arr[2]] = arr[1]
    f.close()
    return dr2

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()



def load_test(img_rows, img_cols, color_type=1):
    print('Read test images')
    path = os.path.join('.', 'dataset' , 'test', '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files)/10)
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im(fl, img_rows, img_cols, color_type)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    return X_test, X_test_id


def cache_data(data, path):
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        print('Restore data from pickle........')
        file = open(path, 'rb')
        data = pickle.load(file)
    return data


def save_model(model):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'model_json' + '.json'
    model_path = os.path.join('.','cache','model')
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save(model_path)


def read_model(index):
    json_name = 'model' + '.json'
    weight_name = 'model_weights' + '.h5'
    model = model_from_json(open(os.path.join('cache', json_name)).read())
    model.load_weights(os.path.join('cache', weight_name))
    return model


def split_validation_set(train, target, test_size):
    random_state = 51
    X_train, X_test, y_train, y_test = \
        train_test_split(train, target,
                         test_size=test_size,
                         random_state=random_state)
    return X_train, X_test, y_train, y_test


def read_and_normalize_and_shuffle_train_data(img_rows, img_cols, batch,
                                              color_type=1):

    cache_path = os.path.join('.','cache', 'train_r_' + str(img_rows) +
                              '_c_' + str(img_cols) + '_t_' +
                              str(color_type)+'_batch_'+ str(batch) + '.dat')

    if not os.path.isfile(cache_path) or use_cache == 0:
        train_data, train_target, driver_id, unique_drivers = \
            load_train(img_rows, img_cols, batch, color_type)
        cache_data((train_data, train_target, driver_id, unique_drivers),
                   cache_path)
    else:
        print('Restore train from cache!')
        (train_data, train_target, driver_id, unique_drivers) = \
            restore_data(cache_path)
    print(train_data[0])
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
    print('Train shape:', train_data.shape)
    if color_type == 1:
        train_data = train_data.reshape(train_data.shape[0], color_type,
                                        img_rows, img_cols)
    else:
        train_data = train_data.transpose((0, 3, 1, 2))

    train_target = np_utils.to_categorical(train_target, 10)
    train_data = train_data.astype('float32')
    mean_pixel = [103.-939, 116.779, 123.68]
    for c in range(3):
        train_data[:, c, :, :] = train_data[:, c, :, :] - mean_pixel[c]
    train_data /= 255
    perm = permutation(len(train_target))
    train_data = train_data[perm]
    train_target = train_target[perm]
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, driver_id, unique_drivers


def read_and_normalize_test_data(img_rows=64, img_cols=64, color_type=1):
    cache_path = os.path.join('.','cache', 'test_r_' + str(img_rows) +
                              '_c_' + str(img_cols) + '_t_' +
                              str(color_type) + '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        test_data, test_id = load_test(img_rows, img_cols, color_type)
        cache_data((test_data, test_id), cache_path)
    else:
        print('Restore test from cache!')
        (test_data, test_id) = restore_data(cache_path)

    test_data = np.array(test_data, dtype=np.uint8)

    if color_type == 1:
        test_data = test_data.reshape(test_data.shape[0], color_type,
                                      img_rows, img_cols)
    else:
        test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float32')
    mean_pixel = [103.939, 116.779, 123.68]
    for c in range(3):
        test_data[:, c, :, :] = test_data[:, c, :, :] - mean_pixel[c]
    test_data /= 255
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_id

def load_train(img_rows, img_cols, batch, num_batches,color_type=1):
    X_train = []
    y_train = []
    driver_id = []

    train_data = get_driver_data()

    print('Read train images')
    path = os.path.join('.' , 'dataset' , 'train', '*.jpg')
    files = glob.glob(path)
    total_size = len(files)
    start = int(batch*num_batches)
    if batch != 0:
        start = start + 1
    end = int((num_batches) * (batch + 1))
    if end > total_size:
        end = total_size
    i = 0
    l = end-start+1
    print("Loading train images for batch {} with start {} and end {}".format(batch, start, end))
    printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for currentFileIndex in range(start,end):
        fl = files[currentFileIndex]
        flbase = os.path.basename(fl)
        img = get_im(fl, img_rows, img_cols, color_type)
        X_train.append(img)
        y_train.append(int(train_data[flbase][1:]))
        printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
        i = i+1

    return X_train, y_train



class CustomGenerator(keras.utils.Sequence):
    def __init__(self ,x_ids, batch_size = total_batch_size, n_classes = 10, shuffle = True):
        self.x_ids = x_ids
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def on_epoch_end(self):
        self.index = np.arange(self.x_ids)   
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __len__(self):
        return self.x_ids // self.batch_size

    def __getitem__(self, index):
        img_rows = 64
        img_cols = 64
        cache_path = os.path.join('.','cache', 'train_r_' + str(img_rows) +
                                    '_c_' + str(img_cols) + '_t_' +
                                    str(color_type_global)+'_batch_'+ str(index) + '.dat')
        if not os.path.isfile(cache_path) or use_cache == 0:
            train_data, train_target = \
                load_train(img_rows, img_cols, index, self.batch_size, color_type_global)
        else:
            print('Restore train from cache!')
            (train_data, train_target) = \
                restore_data(cache_path)
            return train_data, train_target

        train_data = np.array(train_data, dtype=np.uint8)
        train_target = np.array(train_target, dtype=np.uint8)
        print('Train shape:', train_data.shape)
        if color_type_global == 1:
            train_data = train_data.reshape(train_data.shape[0], color_type_global,
                                            img_rows, img_cols)
        else:
            train_data = train_data.transpose((0, 3, 1, 2))

        train_target = np_utils.to_categorical(train_target, 10)
        train_data = train_data.astype('float32')
        mean_pixel = [103.-939, 116.779, 123.68]
        for c in range(3):
            train_data[:, c, :, :] = train_data[:, c, :, :] - mean_pixel[c]
        train_data /= 255
        perm = permutation(len(train_target))
        train_data = train_data[perm]
        train_target = train_target[perm]
        print('Train shape:', train_data.shape)
        print(train_data.shape[0], 'train samples')
        cache_data((train_data, train_target),
                       cache_path)
        return train_data, train_target


def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret



def vgg_std16_model(img_rows, img_cols, color_type=1):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(color_type,
                                                 img_rows, img_cols)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2),padding='same'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2),padding='same'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2),padding='same'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2),padding='same'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2),padding='same'))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(1000, activation='softmax'))

    # Code above loads pre-trained data and
    # model.layers.pop()
    model.add(Dense(10, activation='softmax'))
    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,  metrics=['accuracy'],loss='categorical_crossentropy')
    return model


def run_cross_validation():

    # Now it loads color image
    # input image dimensions
    img_rows, img_cols = 64, 64
    random_state = 20

    train_len = len(os.listdir('./dataset/train'))
    training_generator = CustomGenerator(train_len)
   
    model = vgg_std16_model(img_rows, img_cols, color_type_global)
    
    print("Now training model......")

    history = model.fit_generator(generator=training_generator, steps_per_epoch = train_len // total_batch_size,
          epochs=15,
          verbose=1,
          workers = 0)
    save_model(model)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


    

        # predictions_valid = model.predict(X_valid, batch_size=128, verbose=1)
        # score = log_loss(Y_valid, predictions_valid)
        # print('Score log_loss: ', score)
        # Store valid predictions
        # for i in range(len(test_index)):
        #    yfull_train[test_index[i]] = predictions_valid[i]
    
    # print('Start testing............')
    # test_data, test_id = read_and_normalize_test_data(img_rows, img_cols,
    #                                                   color_type_global)
    # yfull_test = []

    # for index in range(1, num_fold + 1):
    #     # 1,2,3,4,5
    #     # Store test predictions
    #     model = read_model(index, modelStr)
    #     test_prediction = model.predict(test_data, batch_size=128, verbose=1)
    #     yfull_test.append(test_prediction)

    # info_string = 'loss_' + modelStr \
    #               + '_r_' + str(img_rows) \
    #               + '_c_' + str(img_cols) \
    #               + '_folds_' + str(nfolds) \
    #               + '_ep_' + str(nb_epoch)

    # test_res = merge_several_folds_mean(yfull_test, nfolds)
    # create_submission(test_res, test_id, info_string)


# def test_model_and_submit(start=1, end=1, modelStr=''):
#     img_rows, img_cols = 224, 224
#     # batch_size = 64
#     # random_state = 51
#     nb_epoch = 15

#     print('Start testing............')
#     test_data, test_id = read_and_normalize_test_data(img_rows, img_cols,
#                                                       color_type_global)
#     yfull_test = []

#     for index in range(start, end + 1):
#         # Store test predictions
#         model = read_model(index, modelStr)
#         test_prediction = model.predict(test_data, batch_size=128, verbose=1)
#         yfull_test.append(test_prediction)

#     info_string = 'loss_' + modelStr \
#                   + '_r_' + str(img_rows) \
#                   + '_c_' + str(img_cols) \
#                   + '_folds_' + str(end - start + 1) \
#                   + '_ep_' + str(nb_epoch)

#     test_res = merge_several_folds_mean(yfull_test, end - start + 1)
#     create_submission(test_res, test_id, info_string)

# nfolds, nb_epoch, split

run_cross_validation()

# nb_epoch, split
# run_one_fold_cross_validation(10, 0.1)

# test_model_and_submit(1, 10, 'high_epoch')