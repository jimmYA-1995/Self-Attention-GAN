import tensorflow as tf
from tensorflow.keras import layers


class SpectralNormalization(tf.keras.layers.Wrapper):
    """This wrapper reparameterizes a layer by decoupling the weight's
    magnitude and direction.
    This speeds up convergence by improving the
    conditioning of the optimization problem.
    Weight Normalization: A Simple Reparameterization to Accelerate
    Training of Deep Neural Networks: https://arxiv.org/abs/1602.07868
    Tim Salimans, Diederik P. Kingma (2016)
    WeightNormalization wrapper works for keras and tf layers.
    ```python
      net = WeightNormalization(
          tf.keras.layers.Conv2D(2, 2, activation='relu'),
          input_shape=(32, 32, 3),
          data_init=True)(x)
      net = WeightNormalization(
          tf.keras.layers.Conv2D(16, 5, activation='relu'),
          data_init=True)(net)
      net = WeightNormalization(
          tf.keras.layers.Dense(120, activation='relu'),
          data_init=True)(net)
      net = WeightNormalization(
          tf.keras.layers.Dense(n_classes),
          data_init=True)(net)
    ```
    Arguments:
      layer: a layer instance.
      data_init: If `True` use data dependent variable initialization
    Raises:
      ValueError: If not initialized with a `Layer` instance.
      ValueError: If `Layer` does not contain a `kernel` of weights
      NotImplementedError: If `data_init` is True and running graph execution
    """

    def __init__(self, layer, data_init=True, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)
        self.data_init = data_init
        self._track_trackable(layer, name='layer')
        self._init_critical_section = tf.CriticalSection(name='init_mutex')
        self.is_rnn = isinstance(self.layer, tf.keras.layers.RNN)

        if self.data_init and self.is_rnn:
            logging.warning(
                "WeightNormalization: Using `data_init=True` with RNNs "
                "is advised against by the paper. Use `data_init=False`.")

    def build(self, input_shape):
        """Build `Layer`"""
        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(
            shape=[None] + input_shape[1:])

        if not self.layer.built:
            self.layer.build(input_shape)

        kernel_layer = self.layer.cell if self.is_rnn else self.layer

        if not hasattr(kernel_layer, 'kernel'):
            raise ValueError('`WeightNormalization` must wrap a layer that'
                             ' contains a `kernel` for weights')

        if self.is_rnn:
            kernel = kernel_layer.recurrent_kernel
        else:
            kernel = kernel_layer.kernel

        # The kernel's filter or unit dimension is -1
        self.layer_depth = int(kernel.shape[-1])
        self.kernel_norm_axes = list(range(kernel.shape.rank - 1))

        self.g = self.add_weight(
            name='g',
            shape=(self.layer_depth,),
            initializer='ones',
            dtype=kernel.dtype,
            trainable=True,
            synchronization=tf.VariableSynchronization.AUTO,
            aggregation=tf.compat.v1.VariableAggregation.MEAN
            )
        self.v = kernel

        self._initialized = self.add_weight(
            name='initialized',
            shape=None,
            initializer='zeros',
            dtype=tf.dtypes.bool,
            trainable=False)

        if self.data_init:
            # Used for data initialization in self._data_dep_init.
            with tf.name_scope('data_dep_init'):
                layer_config = tf.keras.layers.serialize(self.layer)
                layer_config['config']['trainable'] = False
                self._naked_clone_layer = tf.keras.layers.deserialize(
                    layer_config)
                self._naked_clone_layer.build(input_shape)
                self._naked_clone_layer.set_weights(self.layer.get_weights())
                if not self.is_rnn:
                    self._naked_clone_layer.activation = None

        self.built = True

    def call(self, inputs):
        """Call `Layer`"""

        def _do_nothing():
            return tf.identity(self.g)

        def _update_weights():
            # Ensure we read `self.g` after _update_weights.
            with tf.control_dependencies(self._initialize_weights(inputs)):
                return tf.identity(self.g)

        g = self._init_critical_section.execute(lambda: tf.cond(
            self._initialized, _do_nothing, _update_weights))

        with tf.name_scope('compute_weights'):
            # Replace kernel by normalized weight variable.
            kernel = tf.nn.l2_normalize(self.v, axis=self.kernel_norm_axes) * g

            if self.is_rnn:
                self.layer.cell.recurrent_kernel = kernel
                update_kernel = tf.identity(self.layer.cell.recurrent_kernel)
            else:
                self.layer.kernel = kernel
                update_kernel = tf.identity(self.layer.kernel)

            # Ensure we calculate result after updating kernel.
            with tf.control_dependencies([update_kernel]):
                outputs = self.layer(inputs)
                return outputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())

    def _initialize_weights(self, inputs):
        """Initialize weight g.
        The initial value of g could either from the initial value in v,
        or by the input value if self.data_init is True.
        """
        with tf.control_dependencies([
                tf.debugging.assert_equal(  # pylint: disable=bad-continuation
                    self._initialized,
                    False,
                    message='The layer has been initialized.')
        ]):
            if self.data_init:
                assign_tensors = self._data_dep_init(inputs)
            else:
                assign_tensors = self._init_norm()
            assign_tensors.append(self._initialized.assign(True))
            return assign_tensors

    def _init_norm(self):
        """Set the weight g with the norm of the weight vector."""
        with tf.name_scope('init_norm'):
            v_flat = tf.reshape(self.v, [-1, self.layer_depth])
            v_norm = tf.linalg.norm(v_flat, axis=0)
            g_tensor = self.g.assign(tf.reshape(v_norm, (self.layer_depth,)))
            return [g_tensor]

    def _data_dep_init(self, inputs):
        """Data dependent initialization."""
        with tf.name_scope('data_dep_init'):
            #print(type(self.g))
            #print(dir(self.g))
            #print(self.g.__class__)
            #print(self.g.__name__)
            # Generate data dependent init values
            x_init = self._naked_clone_layer(inputs)
            data_norm_axes = list(range(x_init.shape.rank - 1))
            m_init, v_init = tf.nn.moments(x_init, data_norm_axes)
            scale_init = 1. / tf.math.sqrt(v_init + 1e-10)

            # RNNs have fused kernels that are tiled
            # Repeat scale_init to match the shape of fused kernel
            # Note: This is only to support the operation,
            # the paper advises against RNN+data_dep_init
            if scale_init.shape[0] != self.g.shape[0]:
                rep = int(self.g.shape[0] / scale_init.shape[0])
                scale_init = tf.tile(scale_init, [rep])

            # Assign data dependent init values
            g_tensor = self.g.assign(self.g * scale_init)
            if hasattr(self.layer, 'bias') and self.layer.bias is not None:
                bias_tensor = self.layer.bias.assign(-m_init * scale_init)
                return [g_tensor, bias_tensor]
            else:
                return [g_tensor]

    def get_config(self):
        config = {'data_init': self.data_init}
        base_config = super(WeightNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def remove(self):
        kernel = tf.Variable(
            tf.nn.l2_normalize(self.v, axis=self.kernel_norm_axes) * self.g,
            name='recurrent_kernel' if self.is_rnn else 'kernel')

        if self.is_rnn:
            self.layer.cell.recurrent_kernel = kernel
        else:
            self.layer.kernel = kernel

        return self.layer

"""class SpectralNormalization(tf.keras.layers.Wrapper):
    \"""This wrapper is modified from 
    https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/layers/wrappers.py
    
    
    Arguments:
      layer: a layer instance.
      data_init: If `True` use data dependent variable initialization
    Raises:
      ValueError: If not initialized with a `Layer` instance.
      ValueError: If `Layer` does not contain a `kernel` of weights
      NotImplementedError: If `data_init` is True and running graph execution
    \"""

    def __init__(self, layer, data_init=True, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)
        self.data_init = data_init
        self._track_trackable(layer, name='layer')
        self._init_critical_section = tf.CriticalSection(name='init_mutex')
        self.is_rnn = isinstance(self.layer, tf.keras.layers.RNN)

        if self.data_init and self.is_rnn:
            logging.warning(
                "WeightNormalization: Using `data_init=True` with RNNs "
                "is advised against by the paper. Use `data_init=False`.")

    def build(self, input_shape):
        \"""Build `Layer`\"""
        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(
            shape=[None] + input_shape[1:])

        if not self.layer.built:
            self.layer.build(input_shape)

        kernel_layer = self.layer.cell if self.is_rnn else self.layer

        if not hasattr(kernel_layer, 'kernel'):
            raise ValueError('`WeightNormalization` must wrap a layer that'
                             ' contains a `kernel` for weights')

        if self.is_rnn:
            kernel = kernel_layer.recurrent_kernel
        else:
            kernel = kernel_layer.kernel

        # The kernel's filter or unit dimension is -1
        self.layer_depth = int(kernel.shape[-1])
        self.temporal_dim = int(tf.reshape(kernel, [-1, self.layer_depth]).shape[0])
        self.kernel_norm_axes = list(range(kernel.shape.rank - 1))
        
        
        self._u = self.add_weight(
            name='u',
            shape=(1,self.layer_depth),
            initializer=tf.keras.initializers.GlorotNormal,
            dtype=kernel.dtype,
            trainable=True)
        self._v = self.add_weight(
            name='g',
            shape=(1,self.temporal_dim),
            initializer=tf.keras.initializers.GlorotNormal,
            dtype=kernel.dtype,
            trainable=True)
        self._u = tf.math.l2_normalize(self._u, axis=1)
        self._v = tf.math.l2_normalize(self._v, axis=1)
        self.v = kernel
        
        \"""self.g = self.add_weight(
            name='g',
            shape=(self.layer_depth,),
            initializer='ones',
            dtype=kernel.dtype,
            trainable=True)

        self._initialized = self.add_weight(
            name='initialized',
            shape=None,
            initializer='zeros',
            dtype=tf.dtypes.bool,
            trainable=False)

        if self.data_init:
            # Used for data initialization in self._data_dep_init.
            with tf.name_scope('data_dep_init'):
                layer_config = tf.keras.layers.serialize(self.layer)
                layer_config['config']['trainable'] = False
                self._naked_clone_layer = tf.keras.layers.deserialize(
                    layer_config)
                self._naked_clone_layer.build(input_shape)
                self._naked_clone_layer.set_weights(self.layer.get_weights())
                if not self.is_rnn:
                    self._naked_clone_layer.activation = None\"""

        self.built = True
    
    def call(self, inputs):
        \"""Call `Layer`\"""

        \"""def _do_nothing():
            return tf.identity(self.g)

        def _update_weights():
            # Ensure we read `self.g` after _update_weights.
            with tf.control_dependencies(self._initialize_weights(inputs)):
                return tf.identity(self.g)

        g = self._init_critical_section.execute(lambda: tf.cond(
            self._initialized, _do_nothing, _update_weights))\"""
        
        with tf.name_scope('compute_weights'):
            # Replace kernel by spectrally normalized weight.
            #with tf.init_scope():
            kernel = self.spectral_normalize()
            
            if self.is_rnn:
                self.layer.cell.recurrent_kernel = kernel
                update_kernel = tf.identity(self.layer.cell.recurrent_kernel)
            else:
                self.layer.kernel = kernel
                update_kernel = tf.identity(self.layer.kernel)

            # Ensure we calculate result after updating kernel.
            with tf.control_dependencies([update_kernel]):
                outputs = self.layer(inputs)
                return outputs
            
    def spectral_normalize(self):
        kernel_mat = tf.reshape(self.v, [self.layer_depth, self.temporal_dim])
        self._v = tf.math.l2_normalize(tf.matmul(self._u, kernel_mat), axis=1)
        update_v = tf.identity(self._v)
        with tf.control_dependencies([update_v]):
            self._u = tf.math.l2_normalize(tf.matmul(self._v, tf.transpose(kernel_mat)), axis=1)
            update_u = tf.identity(self._u)
            with tf.control_dependencies([update_u]): 
                sigma = tf.reduce_sum(tf.matmul(self._u, kernel_mat) * self._v)
                return self.v / sigma

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())

    \"""def _initialize_weights(self, inputs):
        #Initialize weight g.
        #The initial value of g could either from the initial value in v,
        #or by the input value if self.data_init is True.
        
        with tf.control_dependencies([
                tf.debugging.assert_equal(  # pylint: disable=bad-continuation
                    self._initialized,
                    False,
                    message='The layer has been initialized.')
        ]):
            if self.data_init:
                assign_tensors = self._data_dep_init(inputs)
            else:
                assign_tensors = self._init_norm()
            assign_tensors.append(self._initialized.assign(True))
            return assign_tensors

    def _init_norm(self):
        #Set the weight g with the norm of the weight vector.
        with tf.name_scope('init_norm'):
            v_flat = tf.reshape(self.v, [-1, self.layer_depth])
            v_norm = tf.linalg.norm(v_flat, axis=0)
            g_tensor = self.g.assign(tf.reshape(v_norm, (self.layer_depth,)))
            return [g_tensor]

    def _data_dep_init(self, inputs):
        #Data dependent initialization.
        with tf.name_scope('data_dep_init'):
            # Generate data dependent init values
            x_init = self._naked_clone_layer(inputs)
            data_norm_axes = list(range(x_init.shape.rank - 1))
            m_init, v_init = tf.nn.moments(x_init, data_norm_axes)
            scale_init = 1. / tf.math.sqrt(v_init + 1e-10)

            # RNNs have fused kernels that are tiled
            # Repeat scale_init to match the shape of fused kernel
            # Note: This is only to support the operation,
            # the paper advises against RNN+data_dep_init
            if scale_init.shape[0] != self.g.shape[0]:
                rep = int(self.g.shape[0] / scale_init.shape[0])
                scale_init = tf.tile(scale_init, [rep])

            # Assign data dependent init values
            g_tensor = self.g.assign(self.g * scale_init)
            if hasattr(self.layer, 'bias') and self.layer.bias is not None:
                bias_tensor = self.layer.bias.assign(-m_init * scale_init)
                return [g_tensor, bias_tensor]
            else:
                return [g_tensor]\"""

    def get_config(self):
        config = {'data_init': self.data_init}
        base_config = super(WeightNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def remove(self):
        kernel = tf.Variable(
            tf.nn.l2_normalize(self.v, axis=self.kernel_norm_axes) * self.g,
            name='recurrent_kernel' if self.is_rnn else 'kernel')

        if self.is_rnn:
            self.layer.cell.recurrent_kernel = kernel
        else:
            self.layer.kernel = kernel

        return self.layer
\"""

def l2normalize(v, eps=1e-12):
    return tf.math.divide(v,(tf.norm(v) + eps))

class SpectralNormalization(layers.Layer):
    \""" Paper: https://openreview.net/forum?id=B1QRgziT-
        source: https://github.com/pfnet-research/sngan_projection
    \"""

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
        \""" Spectrally Normalized Weight
        \"""
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
"""

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
