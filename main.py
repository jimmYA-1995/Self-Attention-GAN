# -*- coding: utf-8 -*-
import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
from tensorflow.keras import Model, layers, Sequential, datasets, optimizers
from SN_layer import SpectralNormalization

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

DEBUG = False
PATH = 'test'
BATCH_SIZE = 64
LATENT_DIM = 128
EPOCHS = 30
RATIO = 1
DISABLE_LABEL_TRICK = 0
NUM_EXAMPLES = 64

USE_SN = True

def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 127.5 - 1
    y = tf.one_hot(tf.squeeze(y, axis=-1), 10)

    return x, y


def load_datasets(data_size_ratio=1.):
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    train_size = int(x_train.shape[0] * data_size_ratio)
    test_size = int(x_test.shape[0] * data_size_ratio)
    steps_per_epoch = train_size // BATCH_SIZE

    x_train, y_train = x_train[:train_size], y_train[:train_size]
    x_test, y_test = x_test[:test_size], y_test[:test_size]

    print("Xdtype: {}, Ydtype: {}".format(x_train.dtype, y_train.dtype))
    print("Training: x={} y={}, Testing: x={} y={}".format(x_train.shape,
                                                           y_train.shape,
                                                           x_test.shape,
                                                           y_test.shape))

    preprocessed_data = preprocess(x_train, y_train)
    BUFFER_SIZE = x_train.shape[0]

    ds_train = tf.data.Dataset.from_tensor_slices(preprocessed_data)
    ds_train = ds_train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    return ds_train, steps_per_epoch


class Generator(Model):
    def __init__(self):
        super().__init__()
        self.convtr = []
        self.convtr.append(layers.Conv2DTranspose(512, (4, 4), strides=(1, 1)))
        self.convtr.append(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding="SAME"))
        self.convtr.append(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="SAME"))
        self.convtr.append(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="SAME"))
        self.convtr.append(layers.Conv2DTranspose(3, (3, 3), strides=(1, 1), padding="SAME", activation="tanh"))

        if USE_SN:
            self.convtr = [SpectralNormalization(convtr) for convtr in self.convtr]
        else:
            self.bn = [layers.BatchNormalization() for _ in range(4)]

    # @tf.function
    def call(self, inputs, is_training=False):
        x = tf.reshape(inputs, shape=[-1, 1, 1, LATENT_DIM])

        for i in range(len(self.convtr) - 1):
            if USE_SN:
                x = self.convtr[i](x, training=is_training)
            else:
                x = self.convtr[i](x)
                x = self.bn[i](x, training=is_training)
                
            x = layers.LeakyReLU(alpha=0.1)(x)

        x = self.convtr[-1](x)  # bs, 32, 32, 3 (RGB)

        return x


class Discriminator(Model):
    def __init__(self):
        super().__init__()
        self.conv = []
        self.conv.append(layers.Conv2D(64, (3, 3), strides=(1, 1), padding="SAME"))
        self.conv.append(layers.Conv2D(64, (4, 4), strides=(2, 2), padding="SAME"))
        self.conv.append(layers.Conv2D(128, (3, 3), strides=(1, 1), padding="SAME"))
        self.conv.append(layers.Conv2D(128, (4, 4), strides=(2, 2), padding="SAME"))
        self.conv.append(layers.Conv2D(256, (3, 3), strides=(1, 1), padding="SAME"))
        self.conv.append(layers.Conv2D(256, (4, 4), strides=(2, 2), padding="SAME"))
        self.conv.append(layers.Conv2D(512, (3, 3), strides=(1, 1), padding="SAME"))

        if USE_SN:
            self.conv = [SpectralNormalization(conv) for conv in self.conv]
        else:
            self.bn = [layers.BatchNormalization() for _ in range(7)]

        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1, activation='sigmoid')
        if USE_SN:
            self.fc = SpectralNormalization(self.fc)

    # @tf.function
    def call(self, inputs, is_training=False):
        x = inputs
        for i in range(len(self.conv)):
            if USE_SN:
                x = self.conv[i](x, training=is_training)
            else:
                x = self.conv[i](x)
                x = self.bn[i](x, training=is_training)

            x = layers.LeakyReLU(alpha=0.1)(x)

        x = self.flatten(x)
        x = self.fc(x) 

        return x

class Trainer(object):
    def __init__(self):
        self.ds_train, self.steps_per_epoch = load_datasets()
        self.generator = Generator()
        self.discriminator = Discriminator()

        lr_fn_G = tf.optimizers.schedules.ExponentialDecay(2e-4, self.steps_per_epoch, decay_rate=0.99, staircase=True)
        lr_fn_D = tf.optimizers.schedules.ExponentialDecay(2e-4, self.steps_per_epoch * RATIO, decay_rate=0.99, staircase=True)
        self.generator_optimizer = optimizers.Adam(learning_rate=lr_fn_G, beta_1=0.)
        self.discriminator_optimizer = optimizers.Adam(learning_rate=lr_fn_D, beta_1=0.)

        # build model to get target the name of tensors
        self.generator.build(input_shape=(BATCH_SIZE, LATENT_DIM))
        self.var_name_list = [var.name for var in self.generator.trainable_variables]

        # metrics
        self.metrics = {}
        self.metrics['train_Gloss'] = tf.keras.metrics.Mean(
            'generator_loss', dtype=tf.float32)
        self.metrics['train_Dloss'] = tf.keras.metrics.Mean(
            'discriminator_loss', dtype=tf.float32)

        for name in self.var_name_list:
            self.metrics[name] = tf.keras.metrics.Mean(
            name, dtype=tf.float32)

    # Define both loss function
    def generator_loss(self, generated_output):
        return tf.reduce_mean(tf.losses.binary_crossentropy(tf.ones_like(generated_output), generated_output))

    def discriminator_loss(self, real_output, generated_output, flip_label, soft_label):

        if soft_label:
            real_label = tf.random.uniform(real_output.shape, 0.7, 1.2)
            gen_label = tf.random.uniform(generated_output.shape, 0., 0.2)
        else:
            real_label = tf.ones_like(real_output)
            gen_label = tf.zeros_like(generated_output)

        if flip_label:
            a = tf.random.uniform(real_label.shape, 0.0, 1.0)
            indices = tf.where(a < 0.05)
            updates = 1 - tf.gather_nd(real_label, indices)
            real_label = tf.tensor_scatter_nd_update(real_label, indices, updates)

            b = tf.random.uniform(real_label.shape, 0.0, 1.0)
            indices = tf.where(a < 0.05)
            updates = 1 - tf.gather_nd(gen_label, indices)
            gen_label = tf.tensor_scatter_nd_update(gen_label, indices, updates)

        real_loss = tf.reduce_mean(
            tf.losses.binary_crossentropy(real_label, real_output))
        generated_loss = tf.reduce_mean(
            tf.losses.binary_crossentropy(gen_label, generated_output))

        total_loss = real_loss + generated_loss

        return total_loss


    @tf.function
    def train_step(self, images, labels, epoch, print_loss=False):

        # Update D. n times per update of G.
        average_loss = tf.constant(0., dtype=tf.float32)
        for _ in range(RATIO):
            noise = tf.random.normal([labels.shape[0], LATENT_DIM])
            fake_images = self.generator(noise, is_training=True)

            with tf.GradientTape() as disc_tape:
                real_output = self.discriminator(images, is_training=True)
                generated_output = self.discriminator(fake_images, is_training=True)

                # disable label trick after n epochs.
                flip_label = soft_label = True if epoch <= DISABLE_LABEL_TRICK else False
                disc_loss  = self.discriminator_loss(real_output, generated_output, flip_label, soft_label)

                # for computing average loss of this train_step
                average_loss = average_loss + disc_loss
        
            gradients_of_discriminator = disc_tape.gradient(
                disc_loss, self.discriminator.trainable_variables)

            self.discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Update G.
        noise = tf.random.normal([labels.shape[0], LATENT_DIM])
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(noise, is_training=True)
            generated_output = self.discriminator(generated_images, is_training=True)
            gen_loss = self.generator_loss(generated_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        
        # for track gradients of specific tensors
        zip_of_generator = list(zip(gradients_of_generator, self.generator.trainable_variables))
        name_to_grads = {var[1].name: var[0] for var in zip_of_generator}

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        self.metrics['train_Gloss'](gen_loss)
        self.metrics['train_Dloss'](average_loss / RATIO)
        for name in self.var_name_list:
            self.metrics[name](name_to_grads[name])


    def train(self):
        print("Steps per epoch: ",self.steps_per_epoch)
        train_log_dir = 'logs/{}'.format(PATH)
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        for epoch in range(EPOCHS):
            start_time = time.time()
            
            for i, (images, labels) in enumerate(self.ds_train):
                print_loss = True if i == 0 else False
                self.train_step(images, labels, tf.constant(epoch), print_loss=print_loss)

            with train_summary_writer.as_default():
                tf.summary.scalar(
                    'Generator Loss', self.metrics['train_Gloss'].result(), step=epoch)
                tf.summary.scalar(
                    'Discriminator Loss', self.metrics['train_Dloss'].result(), step=epoch)
                for name in self.var_name_list:
                    tf.summary.scalar('grads_norm/{}'.format(name), self.metrics[name].result(), step=epoch)


            # save checkpoints every 20 epochs
            if (epoch+1) % 20 == 0 and not DEBUG:
                print("save checkpoint ...")
                ckpt_path = 'checkpoints/{}/epoch_{}'.format(PATH, epoch)
                self.generator.save_weights(ckpt_path, save_format='tf')
            if (epoch+1) % 5 == 0:
                self.save_sample_images(epoch)

            template = 'Epoch({:.2f} sec): {}, gen_loss: {}, disc_loss: {}'
            print(template.format(time.time()-start_time, epoch+1, self.metrics['train_Gloss'].result(), self.metrics['train_Dloss'].result()))
            sys.stdout.flush()

            self.metrics['train_Gloss'].reset_states()
            self.metrics['train_Dloss'].reset_states()
            for name in self.var_name_list:
                self.metrics[name].reset_states()


    def save_sample_images(self, epoch):

        img_path = os.path.abspath('images/{}/'.format(PATH))
        if not os.path.exists(img_path):
            os.makedirs(img_path)

        sample_img = self.generator(
            random_vector, is_training=False)
        samples = np.uint8(sample_img*127.5+128).clip(0, 255)

        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(8, 8)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample)

        plt.savefig(os.path.join(img_path, 'epoch-{}.png'.format(str(epoch+1).zfill(3))), bbox_inches='tight')
        plt.close(fig)

if __name__ == '__main__':
    # reuse this vector
    random_vector = tf.random.normal([NUM_EXAMPLES, LATENT_DIM])

    trainer = Trainer()
    trainer.train()
