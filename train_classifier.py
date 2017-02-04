from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.misc import imresize
import yaml
import argparse
import logging
import os

import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_utils import to_categorical

from blocklogger import MetaBlockLogger

#LOG_FORMAT = '%(level)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG,
                    format='%(levelname)-8s {} %(message)s'.format(MetaBlockLogger.INDENT_FORMAT))

log = logging.getLogger()
#log.addHandler(logging.StreamHandler())
BL = MetaBlockLogger(log)


def read_numpy_dataset(npz_filename):
    """
    Read the dataset (train and test)
    :param filename: Filename to numpy archive
    :return: Datasets and labels
    """
    with open(npz_filename,'rb') as f:
        try:
            npf = np.load(f)
            return (npf['X'], npf['Y']), (npf['X_test'], npf['Y_test'])
        except Exception as err:
            raise Exception("Could not load numpy dataset") from err


def read_meta(meta_filename):
    """
    Read the dataset (train and test)
    :param meta_filename: Filename to numpy archive
    :return: Meta information about the training
    :rtype: dict
    """
    with open(meta_filename, 'r') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as err:
            raise Exception("Could not load meta file") from err


def vgg16(input, num_class):
    """
    Builds the VGG16 model

    :param input: Input tensor
    :param num_class: Class number to correctly size the last fully connected layer
    :return: tflearn.model
    """

    x = tflearn.conv_2d(input, 64, 3, activation='relu', scope='conv1_1')
    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')

    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_1')
    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')

    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_1')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_2')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool4')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool5')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc6')
    x = tflearn.dropout(x, 0.5, name='dropout1')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc7')
    x = tflearn.dropout(x, 0.5, name='dropout2')

    x = tflearn.fully_connected(x, num_class, activation='softmax', scope='fc8',
                                restore=False)

    return x


def main(pretrained_model_filename,
         dataset_filename,
         meta_filename,
         output_model_dir,
         output_model_name,
         batch_size,
         n_epoch,
         validation_set=0.1):
    with BL("reading meta"):
        meta = read_meta(meta_filename)
        labels = meta["labels"]
        num_classes = len(labels)

    with BL("reading dataset"):
        (X, Y), (X_test, Y_test) = read_numpy_dataset(dataset_filename)
        Y = to_categorical(Y, num_classes)
        Y_test = to_categorical(Y_test, num_classes)

    # Resize and cast to float32 for mean channel preprocessing
    with BL("resizing dataset"):
        X = np.array([imresize(X_item, (224, 224), mode='RGB') for X_item in X]).astype(np.float32)
        X_test = np.array([imresize(X_item, (224, 224), mode='RGB') for X_item in X_test]).astype(np.float32)


    with BL("building model"):
        # Real-time data preprocessing
        img_prep = ImagePreprocessing()
        img_prep.add_featurewise_zero_center(mean=[123.68, 116.779, 103.939],
                                             per_channel=True)

        # Real-time data augmentation
        img_aug = ImageAugmentation()
        img_aug.add_random_flip_leftright()
        img_aug.add_random_rotation(max_angle=25.)

        x = tflearn.input_data(shape=[None, 224, 224, 3],
                               name='input',
                               data_preprocessing=img_prep,
                               data_augmentation=img_aug)

        softmax = vgg16(x, num_classes)

        # Finishes to build the model by running softmax on the last layer
        regression = tflearn.regression(softmax,
                                        optimizer='adam',
                                        loss='categorical_crossentropy',
                                        learning_rate=0.001,
                                        restore=False)

        model = tflearn.DNN(regression,
                            checkpoint_path='vgg-finetuning',
                            max_checkpoints=3,
                            tensorboard_verbose=0,
                            tensorboard_dir="./logs")

    # Loading VGG16 weights into model
    with BL("loading VGG16 weights into model"):
        model.load(pretrained_model_filename, weights_only=True)

    # Loading VGG16 weights into model
    with BL("splitting validation and test"):
        index = int(len(X_test)*validation_set)
        X_validation, Y_validation, X_test, Y_test = X_test[:index], Y_test[:index], X_test[index:], Y_test[index:]

    # Start finetuning
    with BL("training the model"):
        model.fit(X, Y, n_epoch=n_epoch, validation_set=(X_validation, Y_validation), shuffle=True,
                  show_metric=True, batch_size=batch_size, snapshot_epoch=False,
                  snapshot_step=200, run_id='vgg-finetuning')

    with BL("saving the model"):
        output_model_filename = os.path.join(output_model_dir, output_model_name)+'.tfl'
        model.save(output_model_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a tiger/dog classifier over a pretrained VGG16 model')
    parser.add_argument('--pretrained_model_filename', type=str,
                        help='Location of the VGG16 pretrained model', default="resources/vgg16.tflearn")
    parser.add_argument('--dataset_filename', dest='dataset_filename', type=str,
                        help='Location of the tiger/dog dataset numpy archive', default="resources/dataset")
    parser.add_argument('--meta_filename', dest='meta_filename', type=str,
                        help='Location of the meta file', default="resources/meta.yaml")
    parser.add_argument('--output_model_dir', dest='output_model_dir', type=str,
                        help='Directory of the fine-tuned model', default="output/")
    parser.add_argument('--output_model_name', dest='output_model_name', type=str,
                        help='Name of the fine-tuned model', default="tiger_dog")
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        help='Batch size of the training', default=16)
    parser.add_argument('--n_epoch', dest='n_epoch', type=int,
                        help='Max number of epochs of the training', default=50)

    args = parser.parse_args()

    with BL("running the train_clasifier script"):
        main(pretrained_model_filename=args.pretrained_model_filename,
             dataset_filename=args.dataset_filename,
             meta_filename=args.meta_filename,
             output_model_dir=args.output_model_dir,
             output_model_name=args.output_model_name,
             batch_size=args.batch_size,
             n_epoch=args.n_epoch)
