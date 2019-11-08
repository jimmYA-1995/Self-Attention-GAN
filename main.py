# -*- coding: utf-8 -*-
import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
from pprint import pprint
from tensorflow.keras import optimizers
from models import get_generator, get_discriminator
from dataset import get_dataset_and_info

# Define both loss function
def hinge_loss_g(generated_output):
    return -tf.reduce_mean(generated_output)

def hinge_loss_d(real_output, generated_output):
    real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_output))
    generated_loss = tf.reduce_mean(tf.nn.relu(1 + generated_output))
    return real_loss + generated_loss

def cross_entropy_g(generated_output):
    return tf.reduce_mean(tf.losses.binary_crossentropy(tf.ones_like(generated_output), generated_output))

def cross_entropy_d(real_output, generated_output):

    real_loss = tf.reduce_mean(
        tf.losses.binary_crossentropy(tf.ones_like(real_output), real_output))
    generated_loss = tf.reduce_mean(
        tf.losses.binary_crossentropy(tf.zeros_like(generated_output), generated_output))

    total_loss = real_loss + generated_loss

    return total_loss


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.ds_train, self.info = get_dataset_and_info(self.config)
        self.steps_per_epoch = self.info['num_records'] // self.config['batch_size']
        self.generator = get_generator(self.info["num_classes"])
        self.discriminator = get_discriminator(self.info["num_classes"])

        if self.config['loss'] == "cross_entropy":
            self.gloss_fn = cross_entropy_g
            self.dloss_fn = cross_entropy_d
        elif self.config['loss'] == "hinge_loss":
            self.gloss_fn = hinge_loss_g
            self.dloss_fn = hinge_loss_d

        lr_fn_G = tf.optimizers.schedules.ExponentialDecay(1e-4, self.steps_per_epoch, decay_rate=0.99, staircase=True)
        lr_fn_D = tf.optimizers.schedules.ExponentialDecay(4e-4, self.steps_per_epoch * self.config['update_ratio'], decay_rate=0.99, staircase=True)
        self.generator_optimizer = optimizers.Adam(learning_rate=lr_fn_G, beta_1=0.)
        self.discriminator_optimizer = optimizers.Adam(learning_rate=lr_fn_D, beta_1=0.)

        # build model to get target the name of tensors
        self.generator.build(input_shape=[(self.config['batch_size'], self.config['z_dim']), (self.config['batch_size'])])
        self.var_name_list = [var.name for var in self.generator.trainable_variables]

        # metrics
        self.metrics = {}
        self.metrics['G_loss'] = tf.keras.metrics.Mean(
            'generator_loss', dtype=tf.float32)
        self.metrics['D_loss'] = tf.keras.metrics.Mean(
            'discriminator_loss', dtype=tf.float32)

        for name in self.var_name_list:
            self.metrics[name] = tf.keras.metrics.Mean(
            name, dtype=tf.float32)

        self.random_vector = tf.random.normal([config['num_sample'], config['z_dim']])
        self.fix_label = tf.random.uniform((self.config['batch_size'],), 0, self.info['num_classes'], dtype=tf.int32)

    @tf.function
    def train_step(self, images, labels):

        # Update D. n times per update of G.
        average_loss = tf.constant(0., dtype=tf.float32)
        for _ in range(self.config['update_ratio']):
            noise = tf.random.normal([images.shape[0], self.config['z_dim']])
            fake_labels = tf.random.uniform((labels.shape[0],), 0, self.info['num_classes'], dtype=tf.int32)
            generated_images = self.generator([noise, fake_labels], training=True)

            with tf.GradientTape() as disc_tape:
                real_output = self.discriminator([images, labels], training=True)
                generated_output = self.discriminator([generated_images, fake_labels], training=True)
                disc_loss  = self.dloss_fn(real_output, generated_output)

                # for computing average loss of this train_step
                average_loss = average_loss + disc_loss
        
            gradients_of_discriminator = disc_tape.gradient(
                disc_loss, self.discriminator.trainable_variables)

            self.discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Update G.
        noise = tf.random.normal([labels.shape[0], self.config['z_dim']])
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator([noise, fake_labels], training=True)
            generated_output = self.discriminator([generated_images, fake_labels], training=True)
            gen_loss = self.gloss_fn(generated_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        
        # for track gradients of specific tensors
        zip_of_generator = list(zip(gradients_of_generator, self.generator.trainable_variables))
        name_to_grads = {var[1].name: var[0] for var in zip_of_generator}

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        self.metrics['G_loss'](gen_loss)
        self.metrics['D_loss'](average_loss / self.config['update_ratio'])
        for name in self.var_name_list:
            self.metrics[name](name_to_grads[name])


    def train(self):
        # print("Steps per epoch: ",self.steps_per_epoch)
        log_dir = 'logs/{}'.format(self.config['path_root'])

        for epoch in range(self.config['epoch']):
            start_time = time.time()
            
            for i, (images, labels) in enumerate(self.ds_train):
                self.train_step(images, labels)

            with tf.summary.create_file_writer(log_dir).as_default():
                tf.summary.scalar(
                    'Generator Loss', self.metrics['G_loss'].result(), step=epoch)
                tf.summary.scalar(
                    'Discriminator Loss', self.metrics['D_loss'].result(), step=epoch)
                for name in self.var_name_list:
                    tf.summary.scalar('grads_norm/{}'.format(name), self.metrics[name].result(), step=epoch)


            # save checkpoints every 20 epochs
            if (epoch+1) % 10 == 0 and not self.config['debug']:
                print("save checkpoint ...")
                ckpt_path = 'checkpoints/{}/epoch_{}'.format(self.config['path_root'], epoch)
                self.generator.save_weights(ckpt_path, save_format='tf')
            if (epoch+1) % 5 == 0:
                self.save_sample_images(epoch)

            template = 'Epoch({:.2f} sec): {}, gen_loss: {}, disc_loss: {}'
            print(template.format(time.time()-start_time, epoch+1, self.metrics['G_loss'].result(), self.metrics['D_loss'].result()))
            sys.stdout.flush()

            self.metrics['G_loss'].reset_states()
            self.metrics['D_loss'].reset_states()
            for name in self.var_name_list:
                self.metrics[name].reset_states()


    def save_sample_images(self, epoch):

        img_path = os.path.abspath('images/{}/'.format(self.config['path_root']))
        if not os.path.exists(img_path):
            os.makedirs(img_path)

        sample_img = self.generator(
            [self.random_vector, self.fix_label], training=False)
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

def main(config):
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    # Handle cuDNN failure issue.
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    parser = argparse.ArgumentParser(description='Experiment parameters')
    parser.add_argument("--debug", action='store_true', default=False,
                        help="whether to use debug mode")
    parser.add_argument("--path_root", default='test',
                        help="path root of images, checkpoints, and logs")
    parser.add_argument("--data_path", default='/home/yct/data/ILSVRC2017_CLS-LOC/ILSVRC/Data/CLS-LOC', help="path to the dataset")
    parser.add_argument("--z_dim", type=int, default=128,
                        help="dimension of noise")
    parser.add_argument('-b', "--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument('-l', "--loss", default="hinge_loss",
                        help="loss function")
    parser.add_argument('-e', '--epoch', type=int, default=5,
                        help="training epochs")
    parser.add_argument('-u', '--update_ratio', type=int, default=1,
                        help="updating ratio of Discriminator to Generator")
    parser.add_argument('-d', '--data_size', type=int, default=-1,
                        help="data size. -1 means full data")
    parser.add_argument('-n', '--num_sample', type=int, default=64,
                        help='the num of samples Generator creates')
    args, unknown = parser.parse_known_args()

    config = {attr: getattr(args, attr) for attr in dir(args) if attr[0]!='_'}
    pprint(config)

    main(config)


