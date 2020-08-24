from __future__ import absolute_import


import os
import sys
import numpy as np
import tensorflow as tf

from models import get_generator
from dataset import get_dataset_and_info

LATENT_DIM = 128

def test_generator():
    filepath = '/home/yct/data/imagenet_small'
    dataset, info = get_dataset_and_info(filepath)
    sample = next(iter(dataset.take(1)))
    batch_size = sample[0].shape[0]
    num_classes = info['num_classes']

    generator = get_generator(num_classes)
    z = tf.random.normal([batch_size, LATENT_DIM])
    label = tf.cast(np.random.randint(0, num_classes, (batch_size)), tf.int32)
    fake_img = generator([z, label])
    
    assert fake_img.shape == [batch_size, 128, 128, 3], fake_img.shape
    print("Generator OK")
