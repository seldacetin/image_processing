import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def create_train_data(train_data_dir, num_classes, resize_col = 64, resize_row = 64, test_size = 0.2, random_state = 1):
    data = []
    labels = []

    train_files = os.listdir(train_data_dir)

    for classes in train_files:
        classname = str(classes)
        path = os.path.join(train_data_dir, classname)
        images = os.listdir(path)
        

        for image in images:
            image_path = os.path.join(path, image)
            image_array = cv2.imread(image_path)
            image_array = cv2.resize(image_array, (resize_row, resize_col))
            data.append(image_array)
            labels.append(classname)

        data = np.array(data)
        labels = np.array(labels)

    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size = test_size, random_state = random_state)
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)

    return x_train, y_train, x_val, y_val 


def create_test_data(dataset_dir, test_info_dir, num_classes, resize_col = 64, resize_row = 64):
    y_test = pd.read_csv(test_info_dir)
    images = y_test['Path'].values
    y_test = y_test["ClassId"].values

    x_train.shape
    x_test.shape
    x_train = []
    x_test = []

    for image in images:
        image_path = os.path.join(dataset_dir, image)
        image_array = cv2.imread(image_path)
        image_array = cv2.resize(image_array, (resize_row, resize_col))
        x_test.append(image_array)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    y_test = to_categorical(y_test, num_classes)

    return x_test, y_test 


print(x_train.shape)






"""
def create_test_data(dataset_dir, test_info_dir, num_classes, resize_col = 64, resize_row = 64):
    y_test = pd.read_csv(test_info_dir)
    images = y_test['Path'].values
    y_test = y_test["ClassId"].values

    x_train = []
    x_test = []
    
    for image in images:
        image_path = os.path.join(dataset_dir, image)
        image_array = cv2.imread(image_path)
        image_array = cv2.resize(image_array, (resize_row, resize_col))
        x_test.append(image_array)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    y_test = to_categorical(y_test, num_classes)
    
    return x_test, y_test
x_test, y_test = create_test_data(dataset_dir, test_info_dir, num_classes, resize_col, resize_row)
print(x_test.shape)

"""