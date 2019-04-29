import numpy as np
from time import strftime, localtime
import sys


def get_time():
    return strftime("%Y-%m-%d %H:%M:%S", localtime())


def unison_shuffle(X, y):
    assert X.shape[0] == y.shape[0]
    p = np.random.permutation(X.shape[0])
    return X[p], y[p]


def save_matrices_to_disk(
        X, y, split, saves_directory, prefix, num_datapoints):
    num_total = X.shape[0]
    num_test = int(num_total * split[1])
    num_val = int(num_total * split[0])
    num_train = num_total - num_test - num_val
    print('number of total examples is ', num_total)
    print('indexes for splitting between train/val/test are ',
          [num_train, num_train + num_val])
    X_train, X_val, X_test = np.split(X, [num_train, num_train + num_val])
    y_train, y_val, y_test = np.split(y, [num_train, num_train + num_val])
    np.save(
        saves_directory +
        '/' +
        prefix +
        '_X_train_' +
        str(num_datapoints) +
        '.npy',
        X_train)
    np.save(
        saves_directory +
        '/' +
        prefix +
        '_y_train_' +
        str(num_datapoints) +
        '.npy',
        y_train)
    np.save(
        saves_directory +
        '/' +
        prefix +
        '_X_val_' +
        str(num_datapoints) +
        '.npy',
        X_val)
    np.save(
        saves_directory +
        '/' +
        prefix +
        '_y_val_' +
        str(num_datapoints) +
        '.npy',
        y_val)
    np.save(
        saves_directory +
        '/' +
        prefix +
        '_X_test_' +
        str(num_datapoints) +
        '.npy',
        X_test)
    np.save(
        saves_directory +
        '/' +
        prefix +
        '_y_test_' +
        str(num_datapoints) +
        '.npy',
        y_test)


def load_matrices_from_disk(saves_directory, prefix, num_datapoints):
    X_train = np.load(
        saves_directory +
        '/' +
        prefix +
        '_X_train_' +
        str(num_datapoints) +
        '.npy')
    y_train = np.load(
        saves_directory +
        '/' +
        prefix +
        '_y_train_' +
        str(num_datapoints) +
        '.npy')
    X_val = np.load(
        saves_directory +
        '/' +
        prefix +
        '_X_val_' +
        str(num_datapoints) +
        '.npy')
    y_val = np.load(
        saves_directory +
        '/' +
        prefix +
        '_y_val_' +
        str(num_datapoints) +
        '.npy')
    X_test = np.load(
        saves_directory +
        '/' +
        prefix +
        '_X_test_' +
        str(num_datapoints) +
        '.npy')
    y_test = np.load(
        saves_directory +
        '/' +
        prefix +
        '_y_test_' +
        str(num_datapoints) +
        '.npy')
    return X_train, y_train, X_val, y_val, X_test, y_test
