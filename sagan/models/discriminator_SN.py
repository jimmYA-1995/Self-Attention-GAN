import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input, layers
from layers import SpectralNormalization, AttentionLayer


def Block(inputs, output_channels, k, s):
    conv = layers.Conv2D(output_channels, k, s, padding='same')
    x = SpectralNormalization(conv)(inputs)
    x = layers.LeakyReLU(alpha=0.1)(x)

    return x

def get_discriminator(config):
    df_dim = config['df_dim']
    power = np.log2(config['img_size'] / 4).astype('int') # 64->4; 128->5
    img = Input(shape=(config['img_size'], config['img_size'], 3), name='image')
    condition_label = Input(shape=(), dtype=tf.int32, name='condition_label')
    x = img
    
    p=1
    for i in range(7):
        k=3 if i%2==0 else 4
        s=2 if i%2==1 or i==4 else 1
        x = Block(x, df_dim * 2 ** p, k, s)
        if i%2==0:
            p+=1
    #for p in range(power):
    #    x = Block(x, df_dim * 2 ** p)
    #    if config['use_attention'] and int(x.shape[1]) in config['attn_dim_G']:
    #        x = AttentionLayer()(x)

    if config['use_label']:
        x = tf.reduce_sum(x, axis=[1,2])
        outputs = layers.Dense(1)(x)
        embedding = layers.Embedding(config['num_classes'], df_dim * 2 ** (power-1))
        label_feature = SpectralNormalization(embedding)(condition_label)
        outputs += tf.reduce_sum(x * label_feature, axis=1, keepdims=True)
        return Model(inputs=[img, condition_label], outputs=outputs)

    else:
        outputs = layers.Conv2D(1, 3, 1, padding='same')(x)
        return Model(inputs=[img, condition_label], outputs=outputs)

def Optimized_Block(inputs, output_channels):
    conv = layers.Conv2D(output_channels, 3, 1, padding='same')
    x = SpectralNormalization(conv)(inputs)
    x = layers.LeakyReLU(alpha=0.1)(x)

    conv = layers.Conv2D(output_channels, 3, 2, padding='same') # downsample
    x = SpectralNormalization(conv)(x)

    conv = layers.Conv2D(output_channels, 3, 2, padding='same')
    x_ = SpectralNormalization(conv)(inputs)

    return layers.add([x_, x])

def Res_Block(inputs, output_channels, downsample=True):
    stride = 2 if downsample else 1

    x = layers.LeakyReLU(alpha=0.1)(inputs)
    conv = layers.Conv2D(output_channels, 3, 1, padding='same')
    x = SpectralNormalization(conv)(x)
    
    x = layers.LeakyReLU(alpha=0.1)(x)
    conv = layers.Conv2D(output_channels, 3, stride, padding='same')
    x = SpectralNormalization(conv)(x)

    x_ = layers.LeakyReLU(alpha=0.1)(inputs)
    conv = layers.Conv2D(output_channels, 3, stride, padding='same')
    x_ = SpectralNormalization(conv)(x_)

    return layers.add([x_, x])

def get_res_discriminator(config):
    df_dim = config['df_dim']
    img = Input(shape=(config['img_size'], config['img_size'], 3), name='image')
    power = np.log2(config['img_size'] / 4).astype('int')
    condition_label = Input(shape=(), dtype=tf.int32, name='condition_label')

    x = Optimized_Block(img, df_dim * 1) # 64x64
    for p in range(1, power):
        x = Res_Block(x, df_dim * 2 ** p)  # 32x32
        if config['use_attention'] and int(x.shape[1]) in config['attn_dim_G']:
            x = AttentionLayer()(x)
    
    x = Res_Block(x, df_dim * 2 ** power, downsample=False) # 4x4

    
    if config['use_label']:
        x = layers.ReLU()(x)
        x = tf.reduce_sum(x, axis=[1,2])
        outputs = SpectralNormalization(layers.Dense(1))(x)
        embedding = layers.Embedding(config['num_classes'], df_dim * 16)
        label_feature = SpectralNormalization(embedding)(condition_label)
        outputs += tf.reduce_sum(x * label_feature, axis=1, keepdims=True)
        return Model(inputs=[img, condition_label], outputs=outputs)
    else:
        outputs = layers.Conv2D(1, 4, 1, padding='same')(x)
#         outputs = SpectralNormalization(conv)(x)
        return Model(inputs=[img, condition_label], outputs=outputs)
