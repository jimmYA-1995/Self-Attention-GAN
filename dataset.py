import tensorflow as tf
from glob import glob
import pickle
import os

IMAGE_SIZE=128
BUFFER_SIZE = 100000


def get_dataset_from_tfrecord(config):
    filenames = glob(os.path.join(config['data_path'], '*.tfrecords'))
    assert len(filenames) > 0
    raw_dataset = tf.data.TFRecordDataset(filenames)

    image_feature_description = {
    #     'height': tf.io.FixedLenFeature([], tf.int64),
    #     'width': tf.io.FixedLenFeature([], tf.int64),
    #     'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string)
    }

    def _preprocess(example_proto):
        parsed_sample = tf.io.parse_single_example(example_proto, image_feature_description)

        image = tf.io.decode_raw(parsed_sample['image_raw'], tf.uint8)
        image.set_shape(IMAGE_SIZE * IMAGE_SIZE * 3)
        image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
        image = tf.cast(image, tf.float32) * (2. / 255) - 1.
        label = tf.cast(parsed_sample['label'], tf.int32)
        
        return image, label

    dataset = raw_dataset.take(config['data_size']).shuffle(BUFFER_SIZE).map(_preprocess).batch(config['batch_size'], drop_remainder=True)
    return dataset

def get_dataset_and_info(config):
    with open(os.path.join(config['data_path'], 'metadata.pickle'), 'rb') as f:
        info = pickle.load(f)
    print("Load: {} of {} records".format(info['num_records'], config['data_size']))
    dataset = get_dataset_from_tfrecord(config)
    
    return dataset, info

