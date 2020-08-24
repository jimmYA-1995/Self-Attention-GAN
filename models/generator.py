import tensorflow as tf
from tensorflow.keras import Model, Input, Sequential, layers, optimizers
from layers import SpectralNormalization, Attention_Layer


def Block(inputs, output_channels, training=False):
    x = layers.BatchNormalization()(inputs, training=training)
    x = layers.ReLU()(x)
    # no tf.image.resize_nearest_neighbor. Use convtr instead.
    # x = upsample(x)
    convtr = layers.Conv2DTranspose(output_channels, 3, 2, padding='same')
    x = SpectralNormalization(convtr)(x, training=training)
    x = layers.BatchNormalization()(x, training=training)
    x = layers.ReLU()(x)
    conv = layers.Conv2D(output_channels, 3, 1, padding='same')
    x = SpectralNormalization(conv)(x, training=training)

    convtr = layers.Conv2DTranspose(output_channels, 3, 2, padding='same')
    x_ = SpectralNormalization(convtr)(inputs, training=training)

    return layers.add([x_, x])

def get_generator(num_classes, gf_dim=16, training=False):
    z = Input(shape=(128,), name='noisy')
    condition_label = Input(shape=(), dtype=tf.int32, name='condition_label')
    one_hot_label = tf.one_hot(condition_label, depth=num_classes)
    x = layers.Concatenate()([z, one_hot_label])
    x = SpectralNormalization(layers.Dense(4 * 4 * gf_dim * 16))(x)
    x = tf.reshape(x, [-1, 4, 4, gf_dim * 16])

    x = Block(x, gf_dim * 16, training=training) # 8x8
    x = Block(x, gf_dim * 8, training=training)  # 16x16
    x = Block(x, gf_dim * 4, training=training)  # 32x32
    x = Attention_Layer()(x)
    x = Block(x, gf_dim * 2, training=training)  # 64x64
    x = Block(x, gf_dim * 1, training=training)  # 128x128

    x = layers.BatchNormalization()(x, training=training)
    x = layers.ReLU()(x)
    conv = layers.Conv2D(3, 3, 1, padding='same', activation='tanh')
    outputs = SpectralNormalization(conv)(x)

    return Model(inputs=[z, condition_label], outputs=outputs)


