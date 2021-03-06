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
        w = getattr(self.module, self.weight_name)[0]
        height = w.shape[-1]
        width = tf.reshape(w, shape=(height, -1)).shape[1]
        # print("H: ", height, "W: ", width)
        u = tf.random.normal(shape=[1, height])
        v = tf.random.normal(shape=[1, width])
        self.u = l2normalize(u)
        self.v = l2normalize(v)

    def build(self, input_shape):
        self.module.build(input_shape)
        if not self._check_param():
            self._make_param()
        
    def call(self, x, training=False):
        if training is False:
            self.update_uv()
        return self.module.call(x)

    @tf.function
    def update_uv(self):
        """
        Spectrally Normalized Weight
        """
        W = getattr(self.module, self.weight_name)[0]
        W_mat = tf.reshape(W, [W.shape[-1], -1])

        for _ in range(self.Ip):
            self.v = l2normalize(tf.matmul(self.u, W_mat))
            self.u = l2normalize(tf.matmul(self.v, tf.transpose(W_mat)))
            
        sigma = tf.reduce_sum(tf.matmul(self.u, W_mat) * self.v)


        if self.factor:
            sigma = sigma / self.factor

        W =  W / sigma


class Attention_Layer(tf.keras.layers.Layer):
    def __init__(self):
        super(Attention_Layer, self).__init__()

    def build(self, input_shape):
        self.sigma = self.add_weight(shape=(),
                                     initializer='zero',
                                     trainable=True,
                                     name='sigma')
        b, w, h, c = input_shape.as_list()
        self.SN_conv = []
        self.SN_conv.append(SpectralNormalization(layers.Conv2D(c//8, 1, 1)))
        self.SN_conv.append(SpectralNormalization(layers.Conv2D(c // 8, 1, 1)))
        self.SN_conv.append(SpectralNormalization(layers.Conv2D(c // 2, 1, 1)))
        self.SN_conv.append(SpectralNormalization(layers.Conv2D(c, 1, 1)))

        for i in range(len((self.SN_conv)) - 1):
            self.SN_conv[i].build(input_shape)
        
        self.SN_conv[-1].build([b,w,h,c//2])


    def call(self, inputs):
        b, w, h, c = inputs.shape.as_list()
        location_num = w * h
        # downsample_num = location_num // 4

        # phi
        phi = self.SN_conv[0](inputs)
        phi = layers.MaxPool2D(2, 1)(phi)
        phi = tf.reshape(phi, shape=[-1, c//8, location_num]) # already transpose
        
        # theta
        theta = self.SN_conv[1](inputs)
        theta = tf.reshape(theta, [-1, location_num, c//8])

        # attention
        atten = tf.matmul(theta, phi)
        atten = tf.nn.softmax(atten) # [location_num, downsample_num]
        
        # g
        g = self.SN_conv[2](inputs)
        g = layers.MaxPool2D(2, 1)(g)
        g = tf.reshape(g, [-1, location_num, c//2])

        atten_g = tf.matmul(atten, g) # [location_num, c//2]
        atten_g = tf.reshape(atten_g, [-1, w, h, c//2])

        atten_g = self.SN_conv[3](atten_g)
        return layers.add([inputs, self.sigma * atten_g])
        

