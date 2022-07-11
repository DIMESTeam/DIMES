from inits import *
import tensorflow.compat.v1 as tf
import functools
tf.disable_v2_behavior()

flags = tf.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs, weights=None):
        return inputs

    def __call__(self, inputs, weights=None):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs, weights)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, repeat=1, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.0

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.repeat = repeat

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if repeat > 1:
            for l in range(1, repeat):
                with tf.variable_scope(self.name + '_vars_' + str(l)):
                    hidden_dim = output_dim * 2
                    self.vars['weights_1_' + str(l)] = xavier([output_dim, hidden_dim], name='weights_1_' + str(l))
                    self.vars['weights_2_' + str(l)] = xavier([hidden_dim, output_dim], name='weights_2_' + str(l))
                    self.vars['bias_1_' + str(l)] = zeros([hidden_dim], name='bias_' + str(l))

            for l in range(repeat, repeat + 2):
                with tf.variable_scope(self.name + '_vars_' + str(l)):
                    self.vars['weights_1_' + str(l)] = xavier([output_dim, output_dim], name='weights_1_' + str(l))
                    self.vars['bias_1_' + str(l)] = zeros([output_dim], name='bias_' + str(l))

        if self.logging:
            self._log_vars()

    def _call(self, inputs, weights=None):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        if weights is None:
            weights = self.vars

        # transform
        output = dot(x, weights['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += weights['bias']

        if self.repeat > 1:
            output = apply_norm(output)
            for layer in range(1, self.repeat):
                skip_connection = output
                output = dot(output, weights['weights_1_' + str(layer)], sparse=self.sparse_inputs)
                output = tf.nn.relu(output + weights['bias_1_' + str(layer)])
                output = dot(output, weights['weights_2_' + str(layer)], sparse=self.sparse_inputs)
                output = skip_connection + output
                output = apply_norm(output)
            output = 0.02 * output

            for layer in range(self.repeat, self.repeat + 2):
                output = dot(output, weights['weights_1_' + str(layer)], sparse=self.sparse_inputs)
                output = tf.nn.relu(output + weights['bias_1_' + str(layer)])
        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, skip_connection=False, use_glu=False,
                 use_layer_norm=False, mis_bias=False,
                 **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.0

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.skip_connection = skip_connection
        self.use_layer_norm = use_layer_norm
        self.mis_bias = mis_bias
        self.use_glu = use_glu
        scale_init = tf.constant_initializer(value=0.02)
        if use_glu:
            for i in range(len(self.support)):
                scale = tf.get_variable(
                    self.name + '/glu_layernorm_scale_' + str(i), [1],
                    initializer=scale_init, trainable=True)
                self.vars['glu_layernorm_scale_' + str(i)] = tf.maximum(scale, 1e-4)
                self.vars['glu_weights_' + str(i)] = glorot([output_dim, output_dim],
                                                            name='glu_weights_' + str(i))
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

            if not self.sparse_inputs and self.skip_connection:
                self.vars['weights_skip'] = glorot([output_dim, output_dim],
                                                   name='weights_skip')

            if use_layer_norm:
                scale = tf.get_variable(
                    "layernorm_scale", [1], initializer=scale_init, trainable=True)
                self.vars['layernorm_scale'] = tf.maximum(scale, 1e-4)
                if not self.sparse_inputs and self.skip_connection:
                    scale = tf.get_variable(
                        "layernorm_scale2", [1], initializer=scale_init, trainable=True)
                    self.vars['layernorm_scale2'] = tf.maximum(scale, 1e-4)

        if self.logging:
            self._log_vars()

    def _call(self, inputs, weights=None):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        if weights is None:
            weights = self.vars

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, weights['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = weights['weights_' + str(i)]
            if self.use_glu:
                pre_sup_gate = dot(pre_sup, weights['glu_weights_' + str(i)])
                apply_glu_instance_norm = functools.partial(apply_norm,
                                                            scale=weights['glu_layernorm_scale_' + str(i)])
                pre_sup_gate = 1.0 + 0.1 * apply_glu_instance_norm(pre_sup_gate)
                pre_sup = tf.sigmoid(pre_sup_gate) * pre_sup
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += weights['bias']

        if self.mis_bias:
            new_support = tf.SparseTensor(indices=self.support[1].indices,
                                          values=tf.ones_like(self.support[1].values),
                                          dense_shape=self.support[1].dense_shape)
            pseudo_output = tf.reshape(output, [-1, output.shape[-1] // 2, 2])
            pseudo_output = tf.concat([tf.zeros_like(pseudo_output[:, :, :1]),
                                       tf.ones_like(pseudo_output[:, :, :1])], axis=-1)
            pseudo_output = tf.reshape(pseudo_output, [-1, output.shape[-1]])
            degree_bias = dot(new_support, pseudo_output, sparse=True)
            output += - 1.0 * (degree_bias / (1 + tf.reduce_max(degree_bias, axis=0, keepdims=True)))

        if self.sparse_inputs or not self.skip_connection:
            if self.use_layer_norm:
                apply_instance_norm = functools.partial(apply_norm,
                                                        scale=weights['layernorm_scale'])
                return apply_instance_norm(self.act(output))
            else:
                return self.act(output)
        else:
            output = dot(self.act(output), weights['weights_skip'])
            if self.use_layer_norm:
                apply_instance_norm = functools.partial(apply_norm,
                                                        scale=weights['layernorm_scale'])
                apply_instance_norm2 = functools.partial(apply_norm,
                                                         scale=weights['layernorm_scale2'])
                return apply_instance_norm(output) + apply_instance_norm2(inputs)
            else:
                return output + inputs


def apply_norm(x, scale=None, epsilon=1e-20):
    axes = [-1]
    mean = tf.reduce_mean(x, axis=axes, keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=axes, keepdims=True)
    result = (x - mean) * tf.rsqrt(variance + epsilon)
    if scale is not None:
        result = result * scale
    return result
