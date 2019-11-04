import sys
import os
import tensorflow as tf
from test import test_generator, test_discriminator

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

test_generator()
test_discriminator()