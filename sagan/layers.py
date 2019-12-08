import tensorflow as tf
from tensorflow.keras import layers
        
def l2normalize(v, eps=1e-12):
    return tf.math.divide(v,(tf.norm(v) + eps))

class SpectralNormalization(layers.Layer):
    """ Paper: https://openreview.net/forum?id=B1QRgziT-
        source: https://github.com/pfnet-research/sngan_projection
    """

    def __init__(self, module, name="weights", Ip=1, factor=None):
        super(SpectralNormalization, self).__init__()
        self.module = module
        self.weight_name = name

        if not Ip >= 1:
            raise ValueError("The number of power iterations should be positive integer")
        self.Ip = Ip
        self.factor = factor

    def _check_param(self):
        try:
            u = getattr(self, "u")
            v = getattr(self, "v")
            return True
        except AttributeError:
            return False

    def _make_param(self):
        W = getattr(self.module, self.weight_name)[0]
        height = W.shape[-1]
        width = tf.reshape(W, shape=(height, -1)).shape[1]

        u = tf.random.normal(shape=[1, height])
        v = tf.random.normal(shape=[1, width])
        self.u = l2normalize(u)
        self.v = l2normalize(v)

    def build(self, input_shape):
        self.module.build(input_shape)
        if not self._check_param():
            self._make_param()
        
    def call(self, x, training=None):
        if training:
            self.update_uv()
        return self.module.call(x)

    # @tf.function
    def update_uv(self):
        """ Spectrally Normalized Weight
        """
        W = getattr(self.module, self.weight_name)[0]
        
        with tf.init_scope():
            W_mat = tf.reshape(W, [W.shape[-1], -1])
        
            for _ in range(self.Ip):
                self.v = l2normalize(tf.matmul(self.u, W_mat))
                self.u = l2normalize(tf.matmul(self.v, tf.transpose(W_mat)))
            
            sigma = tf.reduce_sum(tf.matmul(self.u, W_mat) * self.v)

        if self.factor:
            sigma = sigma / self.factor

        W.assign(W / sigma)


class AttentionLayer(layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        # to scaling attention
        self.sigma = self.add_weight(shape=(),
                                     initializer='zero',
                                     trainable=True,
                                     name='sigma')
        b, w, h, c = input_shape.as_list()
        
        self.conv = []
        self.conv.append(layers.Conv2D(c//8, 1, 1))
        self.conv.append(layers.Conv2D(c//8, 1, 1))
        # self.conv.append(layers.Conv2D(c//2, 1, 1))
        self.conv.append(layers.Conv2D(c, 1, 1))

        for i, conv in enumerate(self.conv):
            #if i==len(self.conv)-1:
            #    conv.build([b,w,h,c//2])
            #else:
            conv.build(input_shape)

    def call(self, inputs, training=None):
        b, w, h, c = inputs.shape.as_list()
        location_num = w * h
        downsample_num = location_num // 4

        query = self.conv[0](inputs)
        query = tf.reshape(query, [-1, location_num, c//8])

        key = self.conv[1](inputs)
        key = layers.MaxPool2D(2,2)(key)
        key = tf.reshape(key, [-1, downsample_num, c//8])
        key = tf.transpose(key, [0, 2, 1])

        atten = tf.matmul(query, key)
        atten = tf.nn.softmax(atten, axis=-1) # [location_num, downsample_num]
        
        value = self.conv[2](inputs)
        value = layers.MaxPool2D(2,2)(value)
        value = tf.reshape(value, [-1, downsample_num, c])

        atten_g = tf.matmul(atten, value) # [location_num, c]
        atten_g = tf.reshape(atten_g, [-1, w, h, c])
        # atten_g = self.conv[3](atten_g)
        # return layers.add([(1-self.sigma) * inputs, self.sigma * atten_g])
        return layers.add([inputs, self.sigma * atten_g])
        

