import tensorflow as tf
from tensorflow.keras import Model, Input, Sequential, layers, optimizers
from layers import SpectralNormalization, Attention_Layer


def Optimized_Block(inputs, output_channels, training=False):
    conv = layers.Conv2D(output_channels, 3, 1, padding='same')
    x = SpectralNormalization(conv)(inputs, training=training)
    x = layers.ReLU()(x)

    conv = layers.Conv2D(output_channels, 3, 2, padding='same') # downsample
    x = SpectralNormalization(conv)(x, training=training)

    conv = layers.Conv2D(output_channels, 3, 2, padding='same')
    x_ = SpectralNormalization(conv)(inputs, training=training)

    return layers.add([x_, x])

def Block(inputs, output_channels, downsample=True, training=False):
    stride = 2 if downsample else 1

    x = layers.ReLU()(inputs)
    conv = layers.Conv2D(output_channels, 3, 1, padding='same')
    x = SpectralNormalization(conv)(x, training=training)
    
    x = layers.ReLU()(x)
    conv = layers.Conv2D(output_channels, 3, stride, padding='same')
    x = SpectralNormalization(conv)(x, training=training)

    x_ = layers.ReLU()(inputs)
    conv = layers.Conv2D(output_channels, 3, stride, padding='same')
    x_ = SpectralNormalization(conv)(x_, training=training)

    return layers.add([x_, x])

def get_discriminator(num_classes, df_dim=16, training=False):
    img = Input(shape=(128, 128, 3), name='image')
    condition_label = Input(shape=(), dtype=tf.int32, name='condition_label')

    x = Optimized_Block(img, df_dim * 1, training=training) # 64x64
    x = Block(x, df_dim * 2, training=training)  # 32x32
    x = Attention_Layer()(x)
    
    x = Block(x, df_dim * 4, training=training)  # 16x16
    x = Block(x, df_dim * 8, training=training)  # 8x8
    x = Block(x, df_dim * 16, training=training) # 4x4
    x = Block(x, df_dim * 16, downsample=False, training=training) # 4x4

    x = layers.ReLU()(x)
    x = tf.reduce_sum(x, axis=[1,2])
    
    outputs = SpectralNormalization(layers.Dense(1))(x)
    embedding = layers.Embedding(num_classes, df_dim * 16)
    label_feature = SpectralNormalization(embedding)(condition_label)
    outputs += tf.reduce_sum(x * label_feature, axis=1, keepdims=True)
    
    return Model(inputs=[img, condition_label], outputs=outputs)


