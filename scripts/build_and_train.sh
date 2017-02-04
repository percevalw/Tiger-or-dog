#!/bin/bash
CIFAR_100_DIR='resources/cifar-100-python'
CIFAR_10_DIR='resources/cifar-10-python'
META_FILENAME='resources/meta.yaml'
DATASET_FILENAME='resources/dataset.npz'
PRETRAINED_MODEL_FILENAME='resources/vgg16.tflearn'
OUTPUT_MODEL_DIR='output'
OUTPUT_MODEL_NAME='tiger_dog'
RESOURCES_DIR='resources'
BATCH_SIZE=32
N_EPOCH=100

# Download the pre-trained checkpoint.
if [ ! -f ${PRETRAINED_MODEL_FILENAME} ]; then
  echo 'Downloading the pretrained VGG16 model'
  wget https://dl.dropboxusercontent.com/s/9li9mi4105jf45v/vgg16.tflearn -o ${PRETRAINED_MODEL_FILENAME}
fi

# Download the CIFAR 100 dataset
if [ ! -d "$CIFAR_100_DIR" ]; then
  echo 'Downloading the CIFAR 100 dataset'
  wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
  tar -xvf cifar-100-python.tar.gz
  mv cifar-100-python ${CIFAR_100_DIR}
  rm cifar-100-python.tar.gz
fi

python build_dataset.py \
    --cifar100_directory=${CIFAR_100_DIR} \
    --cifar10_directory=${CIFAR_10_DIR} \
    --dest_filename=${DATASET_FILENAME}

python train_classifier.py \
    --pretrained_model_filename=${PRETRAINED_MODEL_FILENAME} \
    --dataset_filename=${DATASET_FILENAME} \
    --meta_filename=${META_FILENAME} \
    --batch_size=${BATCH_SIZE} \
    --n_epoch=${N_EPOCH} \
    --output_model_dir=${OUTPUT_MODEL_DIR} \
    --output_model_name=${OUTPUT_MODEL_NAME}

