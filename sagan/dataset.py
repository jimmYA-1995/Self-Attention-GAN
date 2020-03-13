import os
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from glob import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator


BUFFER_SIZE = 30000

def get_dataset_from_tfrecord(config):
    print("Get dataset from tfrecords")
    filenames = glob(os.path.join(config['data_path'], '*.tfrecords'))
    assert len(filenames) > 0
    raw_dataset = tf.data.TFRecordDataset(filenames)
    # TODO: check if it can get subset here.

    image_feature_description = {
    #     'height': tf.io.FixedLenFeature([], tf.int64),
    #     'width': tf.io.FixedLenFeature([], tf.int64),
    #     'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string)
    }

    def _preprocess(example_proto):
        img_size = config['img_size']
        parsed_sample = tf.io.parse_single_example(example_proto, image_feature_description)

        image = tf.io.decode_raw(parsed_sample['image_raw'], tf.uint8)
        image.set_shape(img_size * img_size * 3)
        image = tf.reshape(image, [img_size, img_size, 3])
        image = tf.cast(image, tf.float32) * (2. / 255) - 1.
        label = tf.cast(parsed_sample['label'], tf.int64)
        return image, label

    dataset = raw_dataset.take(config['data_size']).shuffle(BUFFER_SIZE).map(_preprocess)
    dataset = dataset.batch(config['global_batch_size'], drop_remainder=True)
    return dataset


class Generator():
    def __init__(self, image_generator):
        self.image_generator = iter(image_generator)
        
    def __call__(self):
        images, labels =  next(self.image_generator)
        images = images * (2. / 255) - 1.
        labels = labels.astype(np.int32)
        return (images, labels)


def get_dataset_from_generator(config):
    print("Get dataset from image generator")
    img_generator = ImageDataGenerator(
               featurewise_center=False,
               samplewise_center=False,
               featurewise_std_normalization=False,
               samplewise_std_normalization=False,
               zca_whitening=False,
               # zca_epsilon=1e-6,
               rotation_range=0,
               width_shift_range=0.3,
               height_shift_range=0.3,
               brightness_range=(0.,255.),
               shear_range=0.,
               zoom_range=0.,
               channel_shift_range=0.,
               fill_mode='nearest',
               cval=0.,
               horizontal_flip=True,
               vertical_flip=False,
               rescale=None,
               preprocessing_function=None,
               data_format=None,
               validation_split=0.0,
               dtype=None)
    
    generator = img_generator.flow_from_directory(
                config['data_path'],
                target_size=(config['img_size'], config['img_size']),
                color_mode='rgb',
                classes=None,
                class_mode='sparse',
                batch_size=config['global_batch_size'],
                shuffle=True,
                seed=None,
                save_to_dir='/root/notebooks/data/generated',
                save_prefix='gen_',
                save_format='png',
                follow_links=False,
                subset=None,
                interpolation='nearest')
#     callable_generator = Generator(generator)
#     sample = callable_generator()
#     print(type(sample[0]), sample[0].shape, sample[0].dtype)
#     print(type(sample[1]), sample[1].shape, sample[1].dtype)
#     ds = tf.data.Dataset.from_generator(
#             callable_generator, 
#             output_types=(np.float32, np.int32))
    def _preprocess(images, labels):
        images = images * (2. / 255) - 1.
        labels = tf.cast(labels, tf.int64)
        return images, labels
    
    ds = tf.data.Dataset.from_tensors(0).repeat().map(lambda _: next(generator)).map(_preprocess).take(config['data_size'])
    return ds


def get_dataset_from_tfds(ds_name, config):
    print("Get dataset from tensorflow dataset")
    data, info = tfds.load(ds_name, with_info=True,
                           in_memory=True, shuffle_files=True)
    assert isinstance(data['train'], tf.data.Dataset)
    config['num_records'] = info.splits['train'].num_examples
    
    def _preprocess(data):
        img_size = tf.constant((config['img_size'], config['img_size']), dtype=tf.int32)
        img = tf.image.resize(data['image'], img_size)
        img = img * (2 / 255.) - 1.
        label = tf.zeros(shape=(), dtype=tf.int64)
        return img, label
    
    dataset = data['train'].take(config['data_size']).map(_preprocess).batch(config['global_batch_size'], drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset, config
    

def get_dataset_and_info(config):
    #if config['dataset'] == 'CelebA':
    #    dataset, config = get_dataset_from_tfds('celeb_a', config)
    with open(os.path.join(config['data_path'], 'metadata.pickle'), 'rb') as f:
        info = pickle.load(f)
    config.update(info)

    print("Load: {} of {} records".format(config['data_size'], config['num_records']))
    if config['use_image_generator']:
        dataset = get_dataset_from_generator(config)
    else:
        dataset = get_dataset_from_tfrecord(config)
    return dataset, config


    