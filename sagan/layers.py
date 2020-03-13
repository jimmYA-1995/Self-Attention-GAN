import tensorflow as tf
from tensorflow.keras import layers
class SpectralNormalization(tf.keras.layers.Wrapper):
    """This wrapper is modified from 
    https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/layers/wrappers.py
    
    
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
        
        self.g = self.add_weight(
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

    def _initialize_weights(self, inputs):
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
