import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import PIL
import time

import tensorflow as tf
import torch


def prepare_torch_dataset(dataset, batch_size, shuffle):
  inputs = np.concatenate((dataset.healthy_images, dataset.covid_images))
  targets = [[1., 0.]] * len(dataset.healthy_images) + [[0., 1.]] * len(dataset.covid_images)

  inputs = torch.Tensor(np.transpose(inputs, (0, 3, 1, 2)))
  targets = torch.Tensor(targets)

  ds = torch.utils.data.TensorDataset(inputs, targets)
  return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2)


def prepare_augmentation(data):
  horizontal_flip = tf.image.flip_left_right
  vertical_flip = tf.image.flip_up_down
  rotation = lambda x: tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=x)

  images = horizontal_flip(data)
  data = np.concatenate((data, images))

  # images = vertical_flip(data)
  # data = np.concatenate((data, images))

  for i in range(-5, 5, 3):
    images = rotation(i).flow(data, shuffle=False)
    images = np.concatenate([images[i] for i in range(len(images))])
    data = np.concatenate((data, images))

  return data


def prepare_images(path, augmentation=False, resize=(256, 256), normalize=True):
  normalization = lambda t: (t-t.min()) / (t.max()-t.min())
  positive_path = path + "covid/"
  negative_path = path + "non/"

  # For augmentation
  files = os.listdir(positive_path)
  positives = np.zeros((len(files), resize[0], resize[1], 3), dtype=np.float32)
  negatives = []

  for i, file in enumerate(files):
    im = np.array(PIL.Image.open(positive_path + file).convert("RGB"))
    if resize: im = tf.image.resize(im, resize).numpy()
    if normalize: im = normalization(im)
    positives[i] = im

  for file in os.listdir(negative_path):
    im = np.array(PIL.Image.open(negative_path + file).convert("RGB"))
    if resize: im = tf.image.resize(im, resize).numpy()
    if normalize: im = normalization(im)
    negatives.append(im)

  if augmentation: positives = prepare_augmentation(positives)
  return positives, negatives
  
  
def prepare_test_images(path, augmentation=False, resize=(256, 256), normalize=True):
  normalization = lambda t: (t-t.min()) / (t.max()-t.min())
  positive_path = path + "covid/"
  negative_path = path + "non/other_diseases/"
  negative_path_2 = path + "non/No_Finding/"

  # For augmentation
  files = os.listdir(positive_path)
  positives = np.zeros((len(files), resize[0], resize[1], 3), dtype=np.float32)
  negatives = []

  for i, file in enumerate(files):
    im = np.array(PIL.Image.open(positive_path + file).convert("RGB"))
    if resize: im = tf.image.resize(im, resize).numpy()
    if normalize: im = normalization(im)
    positives[i] = im

  for file in os.listdir(negative_path):
    im = np.array(PIL.Image.open(negative_path + file).convert("RGB"))
    if resize: im = tf.image.resize(im, resize).numpy()
    if normalize: im = normalization(im)
    negatives.append(im)
    
  for file in os.listdir(negative_path_2):
    im = np.array(PIL.Image.open(negative_path_2 + file).convert("RGB"))
    if resize: im = tf.image.resize(im, resize).numpy()
    if normalize: im = normalization(im)
    negatives.append(im)

  return positives, negatives


def prepare_dataset(path, batch=64, augmentation=False, validation_split=None, test=False, **kwargs):
  # pos, neg = prepare_images(path, augmentation, kwargs['resize'], kwargs['normalize'])
  # inp = np.concatenate([pos, neg])
  # tar = [[0., 1.]] * len(pos) + [[1., 0.]] * len(neg)
  
  inp = np.load(path + "_inputs.npz")['arr_0']
  tar = np.load(path + "_labels.npz")['arr_0']

  if test:
    inputs = tf.data.Dataset.from_tensor_slices(inp)
    labels = tf.data.Dataset.from_tensor_slices(tar)

    dataset = tf.data.Dataset.zip((inputs, labels)).cache().batch(batch)
    return dataset

  elif validation_split:
    val_size = int(len(inp) * validation_split)

    train_inputs = tf.data.Dataset.from_tensor_slices(inp[val_size:])
    train_labels = tf.data.Dataset.from_tensor_slices(tar[val_size:])

    valid_inputs = tf.data.Dataset.from_tensor_slices(inp[:val_size])
    valid_labels = tf.data.Dataset.from_tensor_slices(tar[:val_size])


    train_dataset = tf.data.Dataset.zip((train_inputs, train_labels)).shuffle(10000).cache().batch(batch)
    valid_dataset = tf.data.Dataset.zip((valid_inputs, valid_labels)).cache().batch(batch)

    return train_dataset, valid_dataset

  else:
    inputs = tf.data.Dataset.from_tensor_slices(inp)
    labels = tf.data.Dataset.from_tensor_slices(tar)

    dataset = tf.data.Dataset.zip((inputs, labels)).shuffle(10000).cache().batch(batch)
    return dataset

  
def prepare_dataset_custom_shape(path, batch=64, augmentation=False, **kwargs):
  pos, neg = prepare_images(path, augmentation, kwargs['resize'], kwargs['normalize'])
  inp = np.concatenate([pos, neg])
  tar = [[0., 1.]] * len(pos) + [[1., 0.]] * len(neg)

  inputs = tf.data.Dataset.from_tensor_slices(inp)
  labels = tf.data.Dataset.from_tensor_slices(tar)

  dataset = tf.data.Dataset.zip((inputs, labels)).shuffle(10000).cache().batch(batch)
  return dataset
