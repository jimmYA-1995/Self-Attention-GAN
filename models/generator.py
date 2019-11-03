import tensorflow as tf
from tensorflow.keras import Model, Input, Sequential, layers, optimizers
from layers import SpectralNormalization, Attention_Layer


def Block(inputs, output_channels, is_training=False):
    x = layers.BatchNormalization()(inputs, training=is_training)
    x = layers.ReLU()(x)
    # no tf.image.resize_nearest_neighbor. Use convtr instead.
    # x = upsample(x)
    convtr = layers.Conv2DTranspose(output_channels, 3, 2, padding='same')
    x = SpectralNormalization(convtr)(x, training=is_training)
    x = layers.BatchNormalization()(x, training=is_training)
    x = layers.ReLU()(x)
    conv = layers.Conv2D(output_channels, 3, 1, padding='same')
    x = SpectralNormalization(conv)(x, training=is_training)

    convtr = layers.Conv2DTranspose(output_channels, 3, 2, padding='same')
    x_ = SpectralNormalization(convtr)(inputs, training=is_training)

    return layers.add([x_, x])

def get_generator(num_classes, gf_dim=16, is_training=False):
    z = Input(shape=(128,), name='noisy')
    condition_label = Input(shape=(num_classes,), name='condition_label')
    x = layers.Concatenate()([z, condition_label])
    x = SpectralNormalization(layers.Dense(4 * 4 * gf_dim * 16))(x)
    x = tf.reshape(x, [-1, 4, 4, gf_dim * 16])

    x = Block(x, gf_dim * 16, is_training=is_training) # 8x8
    x = Block(x, gf_dim * 8, is_training=is_training)  # 16x16
    x = Block(x, gf_dim * 4, is_training=is_training)  # 32x32
    x = Attention_Layer()(x)
    x = Block(x, gf_dim * 2, is_training=is_training)  # 64x64
    x = Block(x, gf_dim * 1, is_training=is_training)  # 128x128

    x = layers.BatchNormalization()(x, training=is_training)
    x = layers.ReLU()(x)
    conv = layers.Conv2D(3, 3, 1, padding='same', activation='tanh')
    outputs = SpectralNormalization(conv)(x)

    return Model(inputs=[z, condition_label], outputs=outputs)


