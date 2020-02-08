import pickle
import os
import logging

from os.path import realpath, basename, dirname, isfile, isdir, join, pardir
import numpy as np
import cv2

logger = logging.getLogger(__name__)

def assert_(condition, message="", error=AssertionError):
    """If not condition, log and raise the exception."""
    if not condition:
        logger.error(f'{message}')
        raise error

def imshow(image):
    cv2.imshow('frame', image)
    cv2.waitKey(0)

def v2i(vector):
    if vector.ndim == 1:
        vector = np.expand_dims(vector, 0)
        return _matrix2rgb(vector)[0]
    else:
        return _matrix2rgb(vector)


def _unpickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

def _matrix2rgb(matrix):
    matrix = np.reshape(matrix, (-1, 3, 32, 32))
    matrix = np.stack([matrix[:, 0], matrix[:, 1], matrix[:, 2]], axis=3)
    return matrix

def _get_data(name='data_batch_1', raw=False):
    data_dir  = join(dirname(realpath(__file__)), pardir, 'data')
    data_path = join(data_dir, name)
    assert_(isdir(data_dir), 'Create the directory autoencoder/data', FileNotFoundError)
    assert_(isfile(data_path), f'Could not find {data_path}', FileNotFoundError)
    data = _unpickle(data_path)[b'data']
    if not raw:
        data = _matrix2rgb(data)
    data = data.astype(np.float32) / 256
    return data

def get_train_data(raw=False):
    return _get_data(name='data_batch_1', raw=raw)

def get_valid_data(raw=False):
    return _get_data(name='data_batch_2', raw=raw)

def get_test_data(raw=False):
    return _get_data(name='data_batch_3', raw=raw)

def show_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model accuracy and loss')
    plt.ylabel('Accuracy, Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train acc', 'Val acc', 'Train loss', 'Val loss'], loc='upper left')
    plt.show()

    # # Plot training & validation loss values
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()


if __name__ == '__main__':
    _get_data()