import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.distribute import reduce_util

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
        w = getattr(self.module, self.weight_name)[0]
        w._aggregation = tf.VariableAggregation.MEAN
        height = w.shape[-1]
        width = tf.reshape(w, shape=(height, -1)).shape[1]

        u = tf.random.normal(shape=[1, height])
        v = tf.random.normal(shape=[1, width])        
        self.u = tf.Variable(l2normalize(u), name='sn_u', trainable=False, aggregation=tf.VariableAggregation.MEAN)
        self.v = tf.Variable(l2normalize(v), name='sn_v', trainable=False, aggregation=tf.VariableAggregation.MEAN)
        
    def build(self, input_shape):
        self.module.build(input_shape)
        if not self._check_param():
            self._make_param()
        
    def call(self, x, training=None):
        if training:
            self.update_uv()
        return self.module.call(x)

    #@tf.function
    def update_uv(self):
        """ Spectrally Normalized Weight
        """
        W = getattr(self.module, self.weight_name)[0]
        W_mat = tf.transpose(tf.reshape(W, [-1, W.shape[-1]]), [1, 0])
        u = self.u
        v = self.v

        for _ in range(self.Ip):
            v = l2normalize(tf.matmul(u, W_mat))
            u = l2normalize(tf.matmul(v, tf.transpose(W_mat)))
                       
        sigma = tf.reduce_sum(tf.matmul(u, W_mat) * v)

        if self.factor:
            sigma = sigma / self.factor
        self.u.assign(u)
        self.v.assign(v)
        
        new_value = tf.math.divide(W, sigma)
        W.assign(new_value)
        

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