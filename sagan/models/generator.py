import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input, layers
from layers import SpectralNormalization, AttentionLayer


def Block(inputs, output_channels):
    convtr = layers.Conv2DTranspose(output_channels, 4, 2, padding='same', use_bias=False)
    x = SpectralNormalization(convtr)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    return x

def get_generator(config):
    gf_dim = config['gf_dim']
    z = Input(shape=(config['z_dim'],), name='noisy')
    condition_label = Input(shape=(), dtype=tf.int32, name='condition_label')
    
    if config['use_label']:
        one_hot_label = tf.one_hot(condition_label, depth=config['num_classes'])
        x = layers.Concatenate()([z, one_hot_label])
    else:
        x = z

    x = SpectralNormalization(layers.Dense(4 * 4 * gf_dim * 16))(x)
    x = tf.reshape(x, [-1, 4, 4, gf_dim * 16])

    # to handle different size of images.
    power = np.log2(config['img_size'] / 4).astype('int') # 64->4; 128->5
    
    for p in reversed(range(power)):
        x = Block(x, gf_dim * (2 ** p))
        if config['use_attention'] and int(x.shape[1]) in config['attn_dim_G']:
            x = AttentionLayer()(x)
            
    outputs = layers.Conv2D(3, 4, 1, padding='same', use_bias=False, activation='tanh')(x)
    return Model(inputs=[z, condition_label], outputs=outputs)


def Res_Block(inputs, output_channels):
    x = layers.BatchNormalization()(inputs)
    x = layers.LeakyReLU(alpha=0.1)(x)
    convtr = layers.Conv2DTranspose(output_channels, 3, 2, padding='same', activation='relu', use_bias=False)
    x = SpectralNormalization(convtr)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    convtr = layers.Conv2DTranspose(output_channels, 3, 1, padding='same', activation='relu', use_bias=False)
    x = SpectralNormalization(convtr)(x)

    x_ = layers.BatchNormalization()(inputs)
    x_ = layers.LeakyReLU(alpha=0.1)(x_)
    convtr = layers.Conv2DTranspose(output_channels, 3, 2, padding='same', activation='relu', use_bias=False)
    x_ = SpectralNormalization(convtr)(x_)

    return layers.add([x_, x])

def get_res_generator(config):
    gf_dim = config['gf_dim']
    z = Input(shape=(config['z_dim'],), name='noisy')
    condition_label = Input(shape=(), dtype=tf.int32, name='condition_label')
    if config['use_label']:
        one_hot_label = tf.one_hot(condition_label, depth=num_classes)
        x = layers.Concatenate()([z, one_hot_label])
    else:
        x = z
        
    x = SpectralNormalization(layers.Dense(4 * 4 * gf_dim * 2 ** (power-1)))(x)
    x = tf.reshape(x, [-1, 4, 4, gf_dim * 2 ** (power-1)])
    
    # to handle different size of images.
    power = np.log2(config['img_size'] / 4).astype('int')
    for p in reversed(range(power)):
        x = Res_Block(x, gf_dim * 2 ** p)
        if config['use_attention'] and int(x.shape[1]) in config['attn_dim_G']:
            x = AttentionLayer()(x)

    # x = layers.BatchNormalization()(x)
    # x = layers.ReLU()(x)
    outputs = layers.Conv2D(3, 1, 1, padding='same', activation='tanh')(x)

    return Model(inputs=[z, condition_label], outputs=outputs)
