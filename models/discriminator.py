import tensorflow as tf
from tensorflow.keras import Model, Input, Sequential, layers, optimizers
from layers import SpectralNormalization, Attention_Layer


def Optimized_Block(inputs, output_channels, is_training=False):
    conv = layers.Conv2D(output_channels, 3, 1, padding='same')
    x = SpectralNormalization(conv)(inputs, training=is_training)
    x = layers.ReLU()(x)

    conv = layers.Conv2D(output_channels, 3, 2, padding='same') # downsample
    x = SpectralNormalization(conv)(x, training=is_training)

    conv = layers.Conv2D(output_channels, 3, 2, padding='same')
    x_ = SpectralNormalization(conv)(inputs, training=is_training)

    return layers.add([x_, x])

def Block(inputs, output_channels, downsample=True is_training=False):
    stride = 2 if downsample else 1

    x = layers.ReLU()(inputs)
    conv = layers.Conv2D(output_channels, 3, 1, padding='same')
    x = SpectralNormalization(conv)(x, training=is_training)
    
    x = layers.ReLU()(x)
    conv = layers.Conv2D(output_channels, 3, stride, padding='same')
    x = SpectralNormalization(conv)(x, training=is_training)

    x_ = layers.ReLU()(inputs)
    conv = layers.Conv2D(output_channels, 3, stride, padding='same')
    x_ = SpectralNormalization(conv)(x_, training=is_training)

    return layers.add([x_, x])

def get_discriminator(num_classes, df_dim=16, is_training=False):
    img = Input(shape=(128, 128, 3), name='image')
    condition_label = Input(shape=(num_classes,), name='condition_label')

    x = Optimized_Block(img, df_dim * 1, is_training=is_training) # 64x64
    x = Block(x, df_dim * 2, is_training=is_training)  # 32x32
    x = Attention_Layer()(x)
    
    x = Block(x, df_dim * 4, is_training=is_training)  # 16x16
    x = Block(x, df_dim * 8, is_training=is_training)  # 8x8
    x = Block(x, df_dim * 16, is_training=is_training) # 4x4
    x = Block(x, df_dim * 16, downsample=False, is_training=is_training) # 4x4

    x = layers.ReLU()(x)
    x = tf.reduce_sum(x, axis=[1,2])
    output = SpectralNormalization(Dense(1))(x)

    embedding = layers.Embedding(num_classes, df_dim * 16)
    x_label = SpectralNormalization(embedding)(condition_label)

    output += tf.reduce_sum(x * x_label, axis=1, keepdims=True)
    # conv = layers.Conv2D(3, 3, 1, padding='same', activation='tanh')
    # outputs = SpectralNormalization(conv)(x)

    return Model(inputs=[z, condition_label], outputs=outputs)


