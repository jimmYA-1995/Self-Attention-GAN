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

    # # @tf.function
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


class SNConv2D(tf.keras.layers.Conv2D):
    """Paper: https://openreview.net/forum?id=B1QRgziT-
        source: https://github.com/pfnet-research/sngan_projection
    """

    def __init__(self,
                filters,
                kernel_size,
                strides=(1, 1),
                padding='valid',
                data_format=None,
                dilation_rate=(1, 1),
                activation=None,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                Ip=1,
                factor=None,
                input_shape=None,
                **kwargs):
        
        super(SNConv2D, self).__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.training = None
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
        height = self.w.shape[-1]
        width = tf.reshape(self.w, shape=(height, -1)).shape[1]

        u = tf.random.normal(shape=[1, height])
        v = tf.random.normal(shape=[1, width])
        self.u = l2normalize(u)
        self.v = l2normalize(v)    

    def build(self, input_shape):
        super(SNConv2D, self).build(input_shape)
        self.w = self.add_weight(
            name='sn_conv2d_kernel',
            shape=self.kernel.shape,
            dtype=tf.float32,
            initializer='glorot_uniform',
            #regularizer=None,
            trainable=True,
            #constraint=None,
            #partitioner=None,
            #use_resource=None,
            synchronization=tf.VariableSynchronization.AUTO,
            aggregation=tf.compat.v1.VariableAggregation.MEAN)
        
        if not self._check_param():
            self._make_param()
    
    # @tf.function 
    def call(self, x, training=None):
        """Applies the convolution layer.
        Args:
            x (tensor): Input image.
        Returns:
            tensor: Output of the convolution.
        """
        if training:
            self.update_wuv()    
            
        out = tf.nn.conv2d(
            x, self.w, strides=self.strides, padding='SAME')
        
        if self.bias is not None:
            out += self.bias
        return out
    
    #@tf.function 
    def update_wuv(self):
        with tf.init_scope():
            W_mat = tf.reshape(self.w, [self.w.shape[-1], -1])
            for _ in range(self.Ip):
                self.v = l2normalize(tf.matmul(self.u, W_mat))
                self.u = l2normalize(tf.matmul(self.v, tf.transpose(W_mat)))

            sigma = tf.reduce_sum(tf.matmul(self.u, W_mat) * self.v)

            if self.factor:
                sigma = sigma / self.factor

            self.w.assign(self.w / sigma)        
        

class SNConv2DTranspose(tf.keras.layers.Conv2DTranspose):
    """Paper: https://openreview.net/forum?id=B1QRgziT-
        source: https://github.com/pfnet-research/sngan_projection
    """

    def __init__(self,
                filters,
                kernel_size,
                strides=(1, 1),
                padding='valid',
                output_padding=None,
                data_format=None,
                dilation_rate=(1, 1),
                activation=None,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                Ip=1,
                factor=None,
                input_shape=None,
                **kwargs):
        
        super(SNConv2DTranspose, self).__init__(
            filters,
            kernel_size,
            strides=strides,
            output_padding=output_padding,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        
        self.Ip = Ip
        self.factor = factor
        self.training = None

    def _check_param(self):
        try:
            u = getattr(self, "u")
            v = getattr(self, "v")
            return True
        except AttributeError:
            return False

    def _make_param(self):
        height = self.w.shape[-1]
        width = tf.reshape(self.w, shape=(height, -1)).shape[1]

        u = tf.random.normal(shape=[1, height])
        v = tf.random.normal(shape=[1, width])
        self.u = l2normalize(u)
        self.v = l2normalize(v)    

    def build(self, input_shape):
        super(SNConv2DTranspose, self).build(input_shape)
        self.w = self.add_weight(
            name='sn_conv2dTranspose_kernel',
            shape=self.kernel.shape,
            dtype=tf.float32,
            initializer='glorot_uniform',
            #regularizer=None,
            trainable=True,
            #constraint=None,
            #partitioner=None,
            #use_resource=None,
            synchronization=tf.VariableSynchronization.AUTO,
            aggregation=tf.compat.v1.VariableAggregation.MEAN)

        if not self._check_param():
            self._make_param()

    # @tf.function    
    def call(self, x, training=None):
        b, h, w, c = x.get_shape().as_list()
        if training:
            self.update_wuv()
            
        if self.padding.lower() == 'same':
            nh = h * self.strides[0]
            nw = w * self.strides[1]
        else:
            nh = h + (h - 1) * self.strides[0] + self.w.shape[0] - 1
            nw = w + (w - 1) * self.strides[1] + self.w.shape[1] - 1
        
        out = tf.nn.conv2d_transpose(x, self.w, output_shape=[b, nh, nw, self.w.shape[-2]],
                                     strides=self.strides, padding=self.padding.upper())
        
        if self.bias is not None:
            out += self.bias
        return out

    def update_wuv(self):
        """
        Spectrally Normalized Weight
        """
        with tf.init_scope():
            W_mat = tf.reshape(self.w, [self.w.shape[-1], -1])
            for _ in range(self.Ip):
                self.v = l2normalize(tf.matmul(self.u, W_mat))
                self.u = l2normalize(tf.matmul(self.v, tf.transpose(W_mat)))

            sigma = tf.reduce_sum(tf.matmul(self.u, W_mat) * self.v)

            if self.factor:
                sigma = sigma / self.factor
            
            self.w.assign(self.w / sigma)        
        

class SNDense(tf.keras.layers.Dense):
    """Paper: https://openreview.net/forum?id=B1QRgziT-
        source: https://github.com/pfnet-research/sngan_projection
    """

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 Ip=1,
                 factor=None,
                 **kwargs):

        super(SNDense, self).__init__(units,
                                      activation=activation,
                                      use_bias=use_bias,
                                      kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer,
                                      kernel_regularizer=kernel_regularizer,
                                      bias_regularizer=bias_regularizer,
                                      activity_regularizer=activity_regularizer,
                                      kernel_constraint=kernel_constraint,
                                      bias_constraint=bias_constraint,
                                      **kwargs)
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
        self.w = self.add_weight(
            name='sn_dense_kernel',
            shape=self.weights[0].shape,
            dtype=tf.float32,
            initializer='glorot_uniform',
            #regularizer=None,
            trainable=True,
            #constraint=None,
            #partitioner=None,
            #use_resource=None,
            synchronization=tf.VariableSynchronization.AUTO,
            aggregation=tf.compat.v1.VariableAggregation.MEAN)
        
        W = self.weights[0]
        height = W.shape[-1]
        width = tf.reshape(W, shape=(height, -1)).shape[1]

        u = tf.random.normal(shape=[1, height])
        v = tf.random.normal(shape=[1, width])
        self.u = l2normalize(u)
        self.v = l2normalize(v)    

    def build(self, input_shape):
        super(SNDense, self).build(input_shape)
        if not self._check_param():
            self._make_param()

    # @tf.function    
    def call(self, x, training=None):
        if training:
            with tf.init_scope():
                self.update_wuv()

        out = tf.matmul(x, self.w)

        if self.use_bias:
            out += self.bias
        return out

    def update_wuv(self):
        W_mat = tf.reshape(self.w, [self.w.shape[-1], -1])
        for _ in range(self.Ip):
            self.v = l2normalize(tf.matmul(self.u, W_mat))
            self.u = l2normalize(tf.matmul(self.v, tf.transpose(W_mat)))
            
        sigma = tf.reduce_sum(tf.matmul(self.u, W_mat) * self.v)

        if self.factor:
            sigma = sigma / self.factor

        self.w.assign(self.w / sigma)
        
class AttentionLayer(layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        # to scale attention
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
