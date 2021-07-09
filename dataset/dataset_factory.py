import os
import pandas as pd
import matplotlib.pyplot as plt
import PIL
import numpy as np
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split


class Dataset():
  """
    Dataset class, for representing all possible datasets

    Parameters
    ----------
    self.positive_dir: string
      Directory for COVID images
    
    self.negative_dir: string
      Directory for non-COVID images

    name: string

    valid_split: float, optional
      Percentage of validation split

    from_numpy: bool, optional
      Load dataset from numpy file 
  
    shuffle: bool, optional
      Shuffles the dataset, default: True
  """

  def __init__(self, **kwargs):
    self.normalize = lambda t: (t-t.min()) / (t.max()-t.min())
    self.from_numpy = kwargs.get('from_numpy', False)
    self.valid_split = kwargs.get('valid_split', 0.0)
    self.augmentation = kwargs.get('augmentation', False)
    self.name = kwargs.get('name', "dataset")
    self.shuffle = kwargs.get('shuffle', True)


  def prepare_dataset(self, batch=64):
    inputs = np.concatenate((self.healthy_images, self.covid_images))
    targets = [[1., 0.]] * len(self.healthy_images) + [[0., 1.]] * len(self.covid_images)
    del self.covid_images, self.healthy_images # for decreasing memory usage 

    if self.valid_split:
      train_inputs, valid_inputs, train_labels, valid_labels = train_test_split(inputs, targets,
                                                                                test_size=self.valid_split,
                                                                                random_state=42, shuffle=True)
      train_inputs_data = tf.data.Dataset.from_tensor_slices(train_inputs)
      train_labels_data = tf.data.Dataset.from_tensor_slices(train_labels)
      valid_inputs_data = tf.data.Dataset.from_tensor_slices(valid_inputs)
      valid_labels_data = tf.data.Dataset.from_tensor_slices(valid_labels)
      
      train_dataset = tf.data.Dataset.zip((train_inputs_data, train_labels_data))
      valid_dataset = tf.data.Dataset.zip((valid_inputs_data, valid_labels_data))

      if self.shuffle: 
        train_dataset = train_dataset.shuffle(10000)
        valid_dataset = valid_dataset.shuffle(10000)

      del train_inputs_data, train_labels_data, valid_inputs_data, valid_labels_data # for decreasing memory usage 
      return train_dataset.batch(batch).prefetch(AUTOTUNE), valid_dataset.batch(batch).prefetch(AUTOTUNE)

    else:
      inputs_data = tf.data.Dataset.from_tensor_slices(inputs)
      labels_data = tf.data.Dataset.from_tensor_slices(targets)
      dataset = tf.data.Dataset.zip((inputs_data, labels_data)) 

      if self.shuffle:
        dataset = dataset.shuffle(10000)
      
      del inputs_data, labels_data # for decreasing memory usage 
      return dataset.batch(batch).prefetch(AUTOTUNE)


  def save_dataset(self, as_numpy=False, resize_to=(256,256), augmentation=False, neg_augment=False):
    """
      Saving images either in numpy or png file. Images are normalized before saving.

      Parameters
      ----------
      self.covid_images:   PIL.Image[] (RGB)
      self.healthy_images: PIL.Image[] (RGB)
      self.name:           string 
      as_numpy:            bool, optional (recommended)
      resize_to:           int[], optional
      augmentation:        bool, optional
    """

    print(f">> Saving {self.name} is started ...")
    start_time = time.time()
    
    # Saving raw images 
    if not as_numpy:
      self.positive_path = f"{self.path}/{self.name}/{self.positive_dir}/"
      self.negative_path = f"{self.path}/{self.name}/{self.negative_dir}/"

      if not os.path.exists(f"{self.path}/{self.name}"): os.mkdir(f"{self.path}/{self.name}")
      if not os.path.exists(self.positive_path): os.mkdir(self.positive_path) 
      if not os.path.exists(self.negative_path): os.mkdir(self.negative_path) 

      for i in range(len(self.covid_images)):
        image = self.normalize(np.array(self.covid_images[i])) # Normalization
        plt.imsave(f"{self.positive_path}/{i}.png", image)
        
      for i in range(len(self.healthy_images)):
        image = self.normalize(np.array(self.healthy_images[i])) # Normalization
        plt.imsave(f"{self.negative_path}/{i}.png", image)
    
    # Saving preprocessed images in .npz format
    else: 
      if augmentation:
        self.positive_file_path = f"{self.path}/{self.name}/{self.positive_dir}_aug.npz"
        if neg_augment: 
          self.negative_file_path = f"{self.path}/{self.name}/{self.negative_dir}_aug.npz"
      else:
        self.positive_file_path = f"{self.path}/{self.name}/{self.positive_dir}.npz"
        self.negative_file_path = f"{self.path}/{self.name}/{self.negative_dir}.npz"

      positive_images = np.zeros((len(self.covid_images), resize_to[0], resize_to[1], 3))
      negative_images = np.zeros((len(self.healthy_images), resize_to[0], resize_to[1], 3))
                                  
      for i, image in enumerate(self.covid_images):
        image = self.normalize(np.array(image)) # Normalization
        positive_images[i] = tf.image.resize(image, resize_to, antialias=True)

      for i, image in enumerate(self.healthy_images):
        image = self.normalize(np.array(image)) # Normalization
        negative_images[i] = tf.image.resize(image, resize_to, antialias=True)

      if augmentation: 
        positive_images = self.prepare_augmentation(positive_images)
        if neg_augment: 
          negative_images = self.prepare_augmentation(negative_images)

      np.savez_compressed(self.positive_file_path, positive_images)
      np.savez_compressed(self.negative_file_path, negative_images)

    print(f":: The {self.name} is successfully saved in {time.time() - start_time} sec...")



class Covid5K(Dataset):
  """
    Covid5K Dataset taken from https://github.com/shervinmin/DeepCovid
  """
  def __init__(self, path, train=True, augmentation=False,**kwargs):
    super(Covid5K, self).__init__(**kwargs)
    self.path = path
    self.train = train
    self.augmentation = augmentation

    if self.train: 
      self.positive_dir = "train/covid"
      self.negative_dir = "train/non"
    else:
      self.positive_dir = "test/covid"
      self.negative_dir = "test/non"

    if not self.from_numpy:
      self.covid_images = self.prepare_images(filter = self.positive_dir)
      self.healthy_images = self.prepare_images(filter = self.negative_dir)

    else:
      if self.augmentation:
        self.covid_images = np.load(f"{self.path}/{self.name}/{self.positive_dir}_aug.npz")['arr_0']
      else:
        self.covid_images = np.load(f"{self.path}/{self.name}/{self.positive_dir}.npz")['arr_0']
      self.healthy_images = np.load(f"{self.path}/{self.name}/{self.negative_dir}.npz")['arr_0']

  def prepare_images(self, filter):
    image_path = f"{self.path}/{self.name}/{filter}/"
    return [PIL.Image.open(f"{image_path + i}").convert('RGB') for i in os.listdir(image_path) if not os.path.isdir(f"{image_path + i}")]


  def prepare_augmentation(self, data):
    """
      Augmenting Covid samples by vertical flip, and rotation

      Parameters
      ----------
      data: np.array
    """
    vertical_flip = tf.image.flip_left_right
    rotation = lambda x: tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=x)

    images = vertical_flip(data)
    data = np.concatenate((data, images))

    for i in range(-5, 5, 3):
      images = rotation(i).flow(data, shuffle=False)
      images = np.concatenate([images[i] for i in range(len(images))])
      data = np.concatenate((data, images))

    return data


class ChestXray(Dataset):
  """
    ChestXray Dataset taken from https://github.com/ieee8023/covid-chestxray-dataset
  """
  def __init__(self, path, **kwargs):
    super(ChestXray, self).__init__(**kwargs)
    self.path = path
    self.positive_dir = "COVID-19"
    self.negative_dir = "No Finding"

    if not self.from_numpy:
      if 'metadata.csv' not in os.listdir(f"{self.path}/{self.name}"):
        raise Exception("Metadata couldn't find...")
      if 'images' not in os.listdir(f"{self.path}/{self.name}"):
        raise Exception("Images couldn't find...")
      
      self.meta = pd.read_csv(f"{self.path}/{self.name}/metadata.csv")
      self.image_path = f"{self.path}/{self.name}/images/"

      self.covid_images = self.prepare_images(filter = self.positive_dir)
      self.healthy_images = self.prepare_images(filter = self.negative_dir)

    else:
      self.covid_images = np.load(f"{self.path}/{self.name}/{self.positive_dir}.npz")['arr_0']
      self.healthy_images = np.load(f"{self.path}/{self.name}/{self.negative_dir}.npz")['arr_0']

  def prepare_images(self, filter=None):
    metadata = self.meta
    if filter: metadata = metadata[metadata['finding'].str.contains(filter)]
    metadata = metadata[metadata["folder"] == "images"]
    return [PIL.Image.open(f"{self.image_path + i['filename']}").convert('RGB') for i in metadata.iloc]



class CheXpert(Dataset):
  """
    CheXpert Dataset taken from https://stanfordmlgroup.github.io/competitions/chexpert/
  """
  def __init__(self, path, **kwargs):
    super(CheXpert, self).__init__(**kwargs)
    self.path = path

    self.positive_dir = "Disease"
    self.negative_dir = "No Finding"

    if not self.from_numpy:
      self.covid_images = self.prepare_images(filter = self.positive_dir)
      self.healthy_images = self.prepare_images(filter = self.negative_dir)

    else:
      self.covid_images = np.load(f"{self.path}/{self.name}/{self.positive_dir}.npz")['arr_0']
      self.healthy_images = np.load(f"{self.path}/{self.name}/{self.negative_dir}.npz")['arr_0']

  def prepare_images(self, filter):
    image_path = f"{self.path}/{self.name}/{filter}/"
    return [PIL.Image.open(f"{image_path + i}").convert('RGB') for i in os.listdir(image_path) if not os.path.isdir(f"{image_path + i}")]


class Covid_CT(Dataset):
  """
    Covid-CT Dataset taken from https://github.com/UCSD-AI4H/COVID-CT

    Parameters
    ----------
    split:               string ('train'/'valid'/'test')
    **kwargs
  """
  def __init__(self, path, split='train', **kwargs):

    super(Covid_CT, self).__init__(**kwargs)
    self.path = path
    self.split = split

    self.positive_dir = f"{self.split}/COVID"
    self.negative_dir = f"{self.split}/NonCOVID"
    
    if not self.from_numpy:
      self.covid_images = self.prepare_images(filter = self.positive_dir)
      self.healthy_images = self.prepare_images(filter = self.negative_dir)

    else:
      if self.augmentation: 
        self.covid_images = np.load(f"{self.path}/{self.name}/{self.positive_dir}_aug.npz")['arr_0']
        self.healthy_images = np.load(f"{self.path}/{self.name}/{self.negative_dir}_aug.npz")['arr_0']
      else:
        self.covid_images = np.load(f"{self.path}/{self.name}/{self.positive_dir}.npz")['arr_0']
        self.healthy_images = np.load(f"{self.path}/{self.name}/{self.negative_dir}.npz")['arr_0']

  def prepare_images(self, filter):
    image_path = f"{self.path}/{self.name}/{filter}/"
    return [PIL.Image.open(f"{image_path + i}").convert('RGB') for i in os.listdir(image_path) if not os.path.isdir(f"{image_path + i}")]

  
  def prepare_augmentation(self, images):
    """
      Augmenting Covid samples by horizontal flip, cropping, brightness and contrast

    """
    def random_jitter(image, resize_to=(256, 256)):
      image = tf.image.resize(image, (resize_to[0]+30, resize_to[1]+30), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      return tf.image.random_crop(image, [len(image), resize_to[0], resize_to[1], 3])
    
    perc = int(len(images) * 0.80) # apply augmentation 80% of images randomly

    images = tf.random.shuffle(images)
    vertical = tf.image.random_flip_up_down(images[:perc])

    images = tf.random.shuffle(images)
    jitter = random_jitter(images[:perc])

    images = tf.random.shuffle(images)
    contrast = tf.image.random_contrast(images[:perc], 0.8, 1.)

    # images = tf.random.shuffle(images)
    # brightness = tf.image.random_brightness(images[:perc], max_delta=0.2)
      
    return self.normalize(np.concatenate((images, vertical, jitter, contrast)))


class MergeDataset(Dataset):
  """
    Merge two dataset. 
  """
  def __init__(self, data1, data2, **kwargs):
    super(MergeDataset, self).__init__(**kwargs)
    
    self.covid_images = np.concatenate((data1.covid_images, data2.covid_images))
    self.healthy_images = np.concatenate((data1.healthy_images, data2.healthy_images))
  
  # TODO
  def prepare_custom_split(self):
    pass