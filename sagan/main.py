# -*- coding: utf-8 -*-
import os
import sys
import time
import runpy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pprint import pprint
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from models import get_generator, get_discriminator, get_res_generator, get_res_discriminator
from dataset import get_dataset_and_info
from utils.parameters import get_parameters


# Define both loss function
def hinge_loss_g(disc_output_gen):
    return -tf.reduce_mean(disc_output_gen)

def hinge_loss_d(disc_output_real, disc_output_gen):
    real_loss = tf.reduce_mean(tf.nn.relu(1.0 - disc_output_real))
    generated_loss = tf.reduce_mean(tf.nn.relu(1 + disc_output_gen))
    return real_loss + generated_loss

def cross_entropy_g(disc_output_gen):
    return tf.reduce_mean(
        tf.losses.binary_crossentropy(tf.ones_like(disc_output_gen), disc_output_gen))

def cross_entropy_d(disc_output_real, disc_output_gen):
    real_loss = tf.reduce_mean(
        tf.losses.binary_crossentropy(tf.ones_like(disc_output_real), disc_output_real))
    generated_loss = tf.reduce_mean(
        tf.losses.binary_crossentropy(tf.zeros_like(disc_output_gen), disc_output_gen))

    total_loss = real_loss + generated_loss
    return total_loss


class Trainer(object):
    def __init__(self, config):
        self.ds_train, self.config = get_dataset_and_info(config)
        # ["/gpu:{}".format(i) for i in range(self.config['num_gpu'])]
        self.strategy = tf.distribute.MirroredStrategy() \
                        if self.config['num_gpu'] > 1 \
                        else tf.distribute.OneDeviceStrategy(device="/gpu:0")

        self.steps_per_epoch = self.config['num_records'] // self.config['batch_size']
        print("total steps: ", self.steps_per_epoch * self.config['epoch'])
        
        if config['model'] == 'vanilla':
            self.generator = get_generator(config)
            self.discriminator = get_discriminator(config)
        #TODO: fix resnet model
        #elif config['model'] == 'resnet':
        #    self.generator = get_res_generator(config)
        #    self.discriminator = get_res_discriminator(config)
        else:
            raise ValueError('Unsupported model type')

        if self.config['loss'] == "cross_entropy":
            print("use ce loss")
            self.gloss_fn = cross_entropy_g
            self.dloss_fn = cross_entropy_d
        elif self.config['loss'] == "hinge_loss":
            print("use hinge loss")
            self.gloss_fn = hinge_loss_g
            self.dloss_fn = hinge_loss_d
        else:
            raise ValueError('Unsupported loss type')

        lr_fn_G = ExponentialDecay(self.config['lr_g'],
                                   self.steps_per_epoch,
                                   decay_rate=self.config['decay_rate'],
                                   staircase=True)
        lr_fn_D = ExponentialDecay(self.config['lr_d'],
                                   self.steps_per_epoch * self.config['update_ratio'],
                                   decay_rate=self.config['decay_rate'],
                                   staircase=True)
        self.optimizer_G = optimizers.Adam(learning_rate=lr_fn_G, beta_1=0.)
        self.optimizer_D = optimizers.Adam(learning_rate=lr_fn_D, beta_1=0.)

        # build model & get trainable variables.
        self.generator.build(input_shape=[(self.config['batch_size'], self.config['z_dim']), (self.config['batch_size'])])
        self.discriminator.build(input_shape=[(self.config['batch_size'], config['img_size'], config['img_size'], 3), (self.config['batch_size'])])
        self.generator.summary()
        self.discriminator.summary()
        
        self.var_G = [var.name for var in self.generator.variables]
        self.Train_var_G = [var.name for var in self.generator.trainable_variables]
        self.Train_var_D = [var.name for var in self.discriminator.trainable_variables]
        
        print("-"*20, "generator weights", "-"*20)
        pprint(self.Train_var_G)
        print("-"*20, "discrimiator weights", "-"*20)
        pprint(self.Train_var_D)
        
        # checkpoints
        self.ckpt_G = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer_G, net=self.generator)
        self.ckpt_D = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer_D, net=self.discriminator)
        self.CkptManager_G = tf.train.CheckpointManager(self.ckpt_G, '{}/G'.format(self.config['ckpt_dir']), max_to_keep=10, checkpoint_name='epoch')
        self.CkptManager_D = tf.train.CheckpointManager(self.ckpt_D, '{}/D'.format(self.config['ckpt_dir']), max_to_keep=10, checkpoint_name='epoch')

        # metrics
        self.metrics = {}
        self.metrics['G_loss'] = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)
        self.metrics['D_loss'] = tf.keras.metrics.Mean('discriminator_loss', dtype=tf.float32)
        self.metrics.update({name: tf.keras.metrics.Mean(name, dtype=tf.float32) for name in self.var_G})
        self.metrics.update({name+'/norm': tf.keras.metrics.Mean(name+'/norm', dtype=tf.float32) for name in self.Train_var_G})
        #for name in self.Train_var_G:
        #    self.metrics[name] = 
        #var_name = [var.name for var in self.generator.variables]
        #for name in var_name:
        #    self.metrics[name] = tf.keras.metrics.Mean(
        #    name, dtype=tf.float32)

        self.fixed_vector = tf.random.normal([config['batch_size'], config['z_dim']])
        self.fixed_label = tf.random.uniform((self.config['batch_size'],), 0, self.config['num_classes'], dtype=tf.int32)

    @tf.function
    def train_step(self, images, labels):
        # Update D. n times per update of G.
        accu_loss = tf.constant(0., dtype=tf.float32)
        for _ in range(self.config['update_ratio']):
            noise = tf.random.normal([images.shape[0], self.config['z_dim']])
            fake_labels = tf.random.uniform((labels.shape[0],), 0, self.config['num_classes'], dtype=tf.int32)
            generated_images = self.generator([noise, fake_labels], training=True)

            with tf.GradientTape() as disc_tape:
                disc_output_real = self.discriminator([images, labels], training=True)
                disc_output_gen = self.discriminator([generated_images, fake_labels], training=True)
                disc_loss  = self.dloss_fn(disc_output_real, disc_output_gen)

                # for computing average loss of this train_step
                accu_loss = accu_loss + disc_loss
        
            grads_D = disc_tape.gradient(
                disc_loss, self.discriminator.trainable_variables)

            self.optimizer_D.apply_gradients(
                zip(grads_D, self.discriminator.trainable_variables))

        # Update G.
        noise = tf.random.normal([labels.shape[0], self.config['z_dim']])
        fake_labels = tf.random.uniform((labels.shape[0],), 0, self.config['num_classes'], dtype=tf.int32)
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator([noise, fake_labels], training=True)
            disc_output_gen = self.discriminator([generated_images, fake_labels], training=True)
            gen_loss = self.gloss_fn(disc_output_gen)

        grads_G = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        self.optimizer_G.apply_gradients(zip(grads_G, self.generator.trainable_variables))
        
        # track metrics
        self.metrics['G_loss'](gen_loss)
        self.metrics['D_loss'](accu_loss / self.config['update_ratio'])
        for name, grads_norm in zip(self.Train_var_G, grads_G):
            self.metrics[name+'/norm'](grads_norm)
        for var in self.generator.variables:
            self.metrics[var.name](var)


    def train(self):
        tf.keras.backend.set_learning_phase(True)
        self.summary_writer = tf.summary.create_file_writer(self.config['log_dir'])
        self.total_step = 0
        
        status_G = self.ckpt_G.restore(self.CkptManager_G.latest_checkpoint)
        status_D = self.ckpt_D.restore(self.CkptManager_D.latest_checkpoint)
        
        if self.CkptManager_G.latest_checkpoint:
            status_G.assert_consumed()
            print("Restored Generator from {}".format(self.CkptManager_G.latest_checkpoint))
        if self.CkptManager_D.latest_checkpoint:
            status_D.assert_consumed()
            print("Restored Discriminator from {}".format(self.CkptManager_D.latest_checkpoint))
        if not self.CkptManager_G.latest_checkpoint and not self.CkptManager_D.latest_checkpoint:
            print("Initializing from scratch.")

        for epoch in range(self.config['epoch']):
            self.ckpt_G.step.assign_add(1)
            self.ckpt_D.step.assign_add(1)
            start_time = time.time()
            
            for images, labels in self.ds_train:
                # if self.config['use_image_generator'] and self.total_step % 1562 ==0:
                #     break
                self.train_step(images, labels)
                if self.total_step % self.config['summary_step_freq'] == 0:
                    self.summary_for_image()
                    self.summary_for_var()
                
                self.total_step += 1

            
            with self.summary_writer.as_default():
                tf.summary.scalar(
                    'Generator Loss', self.metrics['G_loss'].result(), step=epoch)
                tf.summary.scalar(
                    'Discriminator Loss', self.metrics['D_loss'].result(), step=epoch)
                for name in self.Train_var_G:
                    tf.summary.scalar('grads_norm/{}'.format(name), self.metrics[name+'/norm'].result(), step=epoch)
                

            template = 'Epoch({:.2f} sec): {}, gen_loss: {}, disc_loss: {}'
            print(template.format(time.time()-start_time, epoch+1, self.metrics['G_loss'].result(), self.metrics['D_loss'].result()))
            sys.stdout.flush()
            
            # save checkpoints & sample images
            if epoch == 5 or int(self.ckpt_G.step) % 10 == 0:
                print("save checkpoint ...")
                _ = self.CkptManager_G.save()
                _ = self.CkptManager_D.save()
                
            if epoch < 5 or (epoch+1) % 5 == 0:
                print("save sample image in image directory")
                self.save_sample_images(epoch)


            self.metrics['G_loss'].reset_states()
            self.metrics['D_loss'].reset_states()
            for name in self.Train_var_G:
                self.metrics[name+'/norm'].reset_states()


    def save_sample_images(self, epoch):
        if not os.path.exists(self.config['img_dir']):
            os.makedirs(self.config['img_dir'])
        
        # 32x32 -> 1x1; 64x64 -> 2x2
        l = self.config['img_size'] // 32
        n = int(np.sqrt(self.config['num_sample']))
        fig = plt.figure(figsize=(l*n, l*n))
        gs = gridspec.GridSpec(l*n, l*n)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(self.samples[:self.config['num_sample']]):
            ax = plt.subplot(gs[i%n*l:(i%n*l+l), i//n*l:(i//n*l+l)])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample)

        plt.savefig(os.path.join(self.config['img_dir'], 'epoch-{}.png'.format(str(epoch+1).zfill(3))), bbox_inches='tight')
        plt.close(fig)

    def summary_for_var(self):
        with self.summary_writer.as_default():
            for name in self.var_G:
                tf.summary.scalar('G_variables/{}'.format(name),
                                  self.metrics[name].result(),
                                  step=self.total_step)

        for name in self.var_G:
            self.metrics[name].reset_states()
            
    def summary_for_image(self):
        sample_img = self.generator([self.fixed_vector, self.fixed_label], training=False)
        self.samples = np.uint8(sample_img*127.5+128).clip(0, 255)
        with self.summary_writer.as_default():
            tf.summary.image("16 training data examples",
                             self.samples[:16],
                             max_outputs=16,
                             step=self.total_step)
            

def main(config):
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    args = get_parameters()
    config_module = runpy.run_path(args.config_path)
    config = config_module.get('config', None)
    if config is None:
        raise RuntimeError("No 'config' in configuration file")

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in config['gpu'])

    # Handle cuDNN failure issue.
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    tf.config.experimental.set_visible_devices([physical_devices[i] for i in config['gpu']], 'GPU')
    config['device'] = tf.config.experimental.list_logical_devices('GPU')
    config['num_gpu'] = len(config['device'])
    config['batch_size'] = config['batch_size_per_gpu'] * config['num_gpu']

    pprint(config)
    main(config)


