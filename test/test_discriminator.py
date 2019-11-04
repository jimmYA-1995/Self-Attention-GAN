from __future__ import absolute_import


import os
import sys
import numpy as np
import tensorflow as tf

from models import get_discriminator
from dataset import get_dataset_and_info

LATENT_DIM = 128

def test_discriminator():
    filepath = '/home/yct/data/imagenet_small'
    dataset, info = get_dataset_and_info(filepath)
    sample = next(iter(dataset.take(1)))
    batch_size = sample[0].shape[0]
    num_classes = info['num_classes']

    discriminator = get_discriminator(num_classes)
    img = tf.random.normal([batch_size, 128, 128, 3])
    label = np.random.randint(0, num_classes, (batch_size))
    label = tf.cast(label, tf.int32)
    
    output = discriminator([img, label])
    
    assert output.shape == [batch_size, 1], output.shape
    print("Discriminator OK")
