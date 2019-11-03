import tensorflow as tf
from glob import glob
import pickle
import os

IMAGE_SIZE=128
BUFFER_SIZE = 1000
BATCH_SIZE = 32


def get_dataset_from_tfrecord(filepath):
    filenames = glob(os.path.join(filepath, '*.tfrecords'))
    assert len(filenames) > 0
    raw_dataset = tf.data.TFRecordDataset(filenames)
    
    image_feature_description = {
    #     'height': tf.io.FixedLenFeature([], tf.int64),
    #     'width': tf.io.FixedLenFeature([], tf.int64),
    #     'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }

    # def _parse_image_function(example_proto):
    #     # Parse the input tf.Example proto using the dictionary above.
    #     parsed_sample = tf.io.parse_single_example(example_proto, image_feature_description)
    #     return parsed_sample['image_raw'], parsed_sample['label']

    # parsed_dataset = raw_dataset.map(_parse_image_function)
    # sample = next(iter(parsed_dataset.take(1)))
    # print(sample)

    def _preprocess(example_proto):
        parsed_sample = tf.io.parse_single_example(example_proto, image_feature_description)

        image = tf.io.decode_raw(parsed_sample['image_raw'], tf.uint8)
        image.set_shape(IMAGE_SIZE * IMAGE_SIZE * 3)
        image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
        image = tf.cast(image, tf.float32) * (2. / 255) - 1.
        label = tf.cast(parsed_sample['label'], tf.int32)
        
        return image, label

    dataset = raw_dataset.shuffle(BUFFER_SIZE).map(_preprocess).batch(BATCH_SIZE)
    return dataset

def get_dataset_and_info(filepath):
    dataset = get_dataset_from_tfrecord(filepath)

    with open(os.path.join(filepath, 'metadata.pickle'), 'rb') as f:
        info = pickle.load(f)
    
    return dataset, info


# def _extract_image_and_label(record):
#   """Extracts and preprocesses the image and label from the record."""
#   features = tf.parse_single_example(
#     record,
#     features={
#       'image_raw': tf.FixedLenFeature([], tf.string),
#       'label': tf.FixedLenFeature([], tf.int64)
#     })
#   image_size = IMAGE_SIZE
#   image = tf.decode_raw(features['image_raw'], tf.uint8)
#   image.set_shape(image_size * image_size * 3)
#   image = tf.reshape(image, [image_size, image_size, 3])

#   image = tf.cast(image, tf.float32) * (2. / 255) - 1.

#   label = tf.cast(features['label'], tf.int32)

#   return image, label

# class InputFunction(object):
#   """Wrapper class that is passed as callable to Estimator."""

#   def __init__(self, is_training, noise_dim, dataset_name, num_classes, data_dir="./dataset",
#                cycle_length=64, shuffle_buffer_size=100000):
#     self.is_training = is_training
#     self.noise_dim = noise_dim
#     split = ('train' if is_training else 'test')
#     self.data_files = tf.gfile.Glob(os.path.join(data_dir, '*.tfrecords'))
#     self.parser = _extract_image_and_label
#     self.num_classes = num_classes
#     self.cycle_length = cycle_length
#     self.shuffle_buffer_size = shuffle_buffer_size

#   def __call__(self, params):
#     """Creates a simple Dataset pipeline."""

#     batch_size = params['batch_size']
#     filename_dataset = tf.data.Dataset.from_tensor_slices(self.data_files)
#     filename_dataset = filename_dataset.shuffle(len(self.data_files))

#     def tfrecord_dataset(filename):
#       buffer_size = 8 * 1024 * 1224
#       return tf.data.TFRecordDataset(filename, buffer_size=buffer_size)

#     dataset = filename_dataset.apply(tf.contrib.data.parallel_interleave(
#         tfrecord_dataset,
#         cycle_length=self.cycle_length, sloppy=True))
#     if self.is_training:
#       dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(
#           self.shuffle_buffer_size, -1))
#     dataset = dataset.map(self.parser, num_parallel_calls=32)
#     dataset = dataset.apply(
#         tf.contrib.data.batch_and_drop_remainder(batch_size))

#     dataset = dataset.prefetch(4)    # Prefetch overlaps in-feed with training
#     images, labels = dataset.make_one_shot_iterator().get_next()
#     labels = tf.squeeze(labels)
#     random_noise = tf.random_normal([batch_size, self.noise_dim])

#     gen_class_logits = tf.zeros((batch_size, self.num_classes))
#     gen_class_ints = tf.multinomial(gen_class_logits, 1)
#     gen_sparse_class = tf.squeeze(gen_class_ints)

#     features = {
#         'real_images': images,
#         'random_noise': random_noise,
#         'fake_labels': gen_sparse_class}

#     return features, labels