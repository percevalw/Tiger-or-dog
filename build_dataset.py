from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from tflearn.datasets import cifar10
from tflearn.data_utils import shuffle, to_categorical
import numpy as np
import pickle
import logging

from blocklogger import MetaBlockLogger

#LOG_FORMAT = '%(asctime)-15s %(message)s'
#LOG_FORMAT = '%(level)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG,
                    format='%(levelname)-8s{} %(message)s'.format(MetaBlockLogger.INDENT_FORMAT))
log = logging.getLogger()
#log.addHandler(logging.StreamHandler())
BL = MetaBlockLogger(log)


def subset(data, count):
    x, y = data
    indices = np.random.choice(len(x), count)
    return x[indices], y[indices]


def unpickle(file):
    fo = open(file, 'rb')
    d = pickle.load(fo, encoding='latin1')
    fo.close()
    return d


def reshape_normalize_cifar_100(d):
    """Helper to have the cifar 100 under the correct shape"""
    return d.reshape((len(d), 3, 32, 32)).swapaxes(3, 1).swapaxes(1, 2) / 255.


def filter_dataset(x, y, old_label, new_label):
    """Filter the dataset to get the specified labeled images only"""
    new_x = x[y == old_label]
    new_y = np.full(len(new_x), new_label)
    return new_x, new_y


def filter_dataset_keep_others(x, y, old_label, new_label, other_label):
    """Splits the dataset to according to the given label"""
    new_x = x[y == old_label]
    new_x_others = x[y != old_label]
    new_y = np.full(len(new_x), new_label)
    new_others_y = np.full(len(new_x_others), other_label)
    return (new_x, new_y), (new_x_others, new_others_y)


def load_cifar_10_batch(fpath):
    with open(fpath, 'rb') as f:
        if sys.version_info > (3, 0):
            # Python3
            d = pickle.load(f, encoding='latin1')
        else:
            # Python2
            d = pickle.load(f)
    data = d["data"]
    labels = d["labels"]
    return data, labels


def load_cifar_10(dirname):
    X_train = []
    Y_train = []

    for i in range(1, 6):
        fpath = os.path.join(dirname, 'data_batch_' + str(i))
        data, labels = load_cifar_10_batch(fpath)
        if i == 1:
            X_train = data
            Y_train = labels
        else:
            X_train = np.concatenate([X_train, data], axis=0)
            Y_train = np.concatenate([Y_train, labels], axis=0)

    fpath = os.path.join(dirname, 'test_batch')
    X_test, Y_test = load_cifar_10_batch(fpath)

    X_train = np.dstack((X_train[:, :1024], X_train[:, 1024:2048],
                         X_train[:, 2048:])) / 255.
    X_train = np.reshape(X_train, [-1, 32, 32, 3])
    X_test = np.dstack((X_test[:, :1024], X_test[:, 1024:2048],
                        X_test[:, 2048:])) / 255.
    X_test = np.reshape(X_test, [-1, 32, 32, 3])

    return (X_train, np.array(Y_train)), (X_test, np.array(Y_test))


def load_cifar_100(dirname):
    meta = unpickle(os.path.join(dirname, 'meta'))

    train = unpickle(os.path.join(dirname, 'train'))

    test = unpickle(os.path.join(dirname, 'test'))

    return (train['data'], np.array(train['fine_labels'])), (test['data'], np.array(test['fine_labels'])), meta


def load_datasets(cifar100_directory, cifar10_directory):
    with BL("loading cifar 100 and cifar 10 dataset"):
        (X_10, Y_10), (X_10_test, Y_10_test) = load_cifar_10(cifar10_directory)

        (X_100_flat, Y_100), (X_100_test_flat, Y_100_test), meta = load_cifar_10(cifar100_directory)

    tiger_label = meta['fine_label_names'].index('tiger')
    dog_label = 5

    with BL("reshaping and normalizing CIFAR100 dataset"):
        X_100 = reshape_normalize_cifar_100(X_100_flat)
        X_100_test = reshape_normalize_cifar_100(X_100_test_flat)

    with BL("picking a subset of the dog samples to balance the dataset"):
        (X_dog, Y_dog) = subset(filter_dataset(X_10, Y_10, dog_label, 0), 1000)
        (X_dog_test, Y_dog_test) = subset(filter_dataset(X_10_test, Y_10_test, dog_label, 0), 600)

        log.info("Training and test dog images extracted : {} training dogs, {} test dogs".format(len(X_dog),
                                                                                                  len(X_dog_test)))

    with BL("splitting cifar100 to get tiger and none samples"):
        (X_tiger, Y_tiger), (X_none_full, Y_none_full) = filter_dataset_keep_others(X_100, Y_100, tiger_label, 1, 2)
        (X_tiger_test, Y_tiger_test), (X_none_full_test, Y_none_full_test) = filter_dataset_keep_others(X_100_test,
                                                                                                        Y_100_test, 88,
                                                                                                        1,
                                                                                                        2)

        X_none, Y_none = subset((X_none_full, Y_none_full), 1000)
        X_none_test, Y_none_test = subset((X_none_full_test, Y_none_full_test), 600)

        log.info(
            "Training and test tiger/none images extracted : {} training tigers, {} training none, {} test tigers, {} test none".format(
                len(X_tiger), len(X_none), len(X_tiger_test), len(X_none_test)))

    with BL("concatenating tiger, dog and none sample to get the final datasets"):
        # We oversample the tiger samples to balance the dataset
        X = np.concatenate((X_tiger, X_tiger, X_dog, X_none))
        Y = np.concatenate((Y_tiger, Y_tiger, Y_dog, Y_none))

        log.info("Training datasets concatanated : {} total".format(len(X)))

        X_test = np.concatenate((X_tiger_test, X_dog_test, X_none_test))
        Y_test = np.concatenate((Y_tiger_test, Y_dog_test, Y_none_test))

        log.info("Test datasets concatanated : {} total".format(len(X_test)))

    with BL("suffling the training dataset"):
        X, Y = shuffle(X, Y)

    return (X, Y), (X_test, Y_test)


def write_numpy_dataset(filename, X, Y, X_test, Y_test):
    with open(filename,'wb') as f:
         np.savez(f, X=X, Y=Y, X_test=X_test, Y_test=Y_test)


def read_numpy_dataset(filename):
    with open(filename,'rb') as f:
        npf = np.load(f)
        return (npf['X'], npf['Y']), (npf['X_test'], npf['Y_test'])


def main(dest_filename, cifar100_directory, cifar10_directory):
    if os.path.exists(dest_filename):
        log.info("{} already exists, skipping dataset build".format(dest_filename))
        return

    with BL("creating the dataset"):
        (X, Y), (X_test, Y_test) = load_datasets(cifar100_directory, cifar10_directory)

    with BL("saving the dataset"):
        write_numpy_dataset(dest_filename, X, Y, X_test, Y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Builds a tiger/dog classifier training/testing dataset')
    parser.add_argument('--dest_filename', dest="dest_filename", type=str,
                        help='Destination name of the dataset numpy archive', default='resources/dataset')
    parser.add_argument('--cifar100_directory', dest='cifar100_directory', type=str,
                        help='Location of the cifar 100 directory', default='resources/cifar-100-python')
    parser.add_argument('--cifar10_directory', dest='cifar10_directory', type=str,
                        help='Location of the cifar 10 directory', default='resources/cifar-10-python')

    args = parser.parse_args()

    with BL("running the build_dataset script"):
        main(dest_filename=args.dest_filename,
             cifar100_directory=args.cifar100_directory,
             cifar10_directory=args.cifar10_directory)
