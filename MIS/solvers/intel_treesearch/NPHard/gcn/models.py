from layers import *
from metrics import *
import reinforce
from layers import _LAYER_UIDS
import tensorflow_addons as tfa
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

flags = tf.flags
FLAGS = flags.FLAGS


def lrelu(x):
  return tf.maximum(x * 0.2, x)


class Model(object):
  def __init__(self, **kwargs):
    allowed_kwargs = {'name', 'logging', 'meta_update', 'meta_steps', 'ppo',
                      'random_heatmap', 'normalize_factor', 'supervision_factor', 'aux_grad'}
    for kwarg in kwargs.keys():
      assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
    name = kwargs.get('name')
    if not name:
      name = self.__class__.__name__.lower()
    self.name = name

    logging = kwargs.get('logging', False)
    self.logging = logging

    self.vars = {}
    self.placeholders = {}

    self.layers = []
    self.activations = []

    self.inputs = None
    self.outputs = None
    self.outputs_softmax = None
    self.pred = None
    self.output_dim = None

    self.loss = 0
    self.accuracy = 0
    self.optimizer = None
    self.opt_op = None
    self.zero_ops = None
    self.accum_ops = None
    self.train_ops = None

  def _build(self):
    raise NotImplementedError

  def build(self):
    """ Wrapper for _build() """
    with tf.variable_scope(self.name):
      self._build()

    # Build sequential layer model
    layer_id = 0
    self.activations.append(self.inputs)
    for layer in self.layers:
      if self.name == 'gcn_deep' and layer_id % 2 == 0 and layer_id > 0 and layer_id < len(self.layers) - 1:
        hidden = layer(self.activations[-1])
        self.activations.append(tf.nn.relu(hidden + self.activations[-2]))
        layer_id = layer_id + 1
      elif layer_id < len(self.layers) - 1:
        hidden = tf.nn.relu(layer(self.activations[-1]))
        self.activations.append(hidden)
        layer_id = layer_id + 1
      else:
        hidden = layer(self.activations[-1])
        self.activations.append(hidden)
        layer_id = layer_id + 1
    self.outputs = self.activations[-1]
    if self.name != 'gcn_dqn':
      self.outputs_softmax = tf.nn.softmax(self.outputs[:, 0:2] / 1e2)
    if self.name == 'gcn_deep_diver':
      for out_id in range(1, FLAGS.diver_num):
        self.outputs_softmax = tf.concat([self.outputs_softmax, tf.nn.softmax(
            self.outputs[:, self.output_dim * out_id:self.output_dim * (out_id + 1)] / 1e2)], axis=1)
    if self.name == 'gcn_dqn':
      self.pred = tf.argmax(self.outputs)
    # Store model variables for easy access
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
    self.vars = {var.name: var for var in variables}

    # Build metrics
    self._loss()
    self._accuracy()

    self.opt_op = self.optimizer.minimize(self.loss)

  def predict(self):
    pass

  def _loss(self):
    raise NotImplementedError

  def _loss_reg(self):
    raise NotImplementedError

  def _accuracy(self):
    raise NotImplementedError

  def save(self, sess=None):
    if not sess:
      raise AttributeError("TensorFlow session not provided.")
    saver = tf.train.Saver(self.vars)
    save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
    print("Model saved in file: %s" % save_path)

  def load(self, sess=None):
    if not sess:
      raise AttributeError("TensorFlow session not provided.")
    saver = tf.train.Saver(self.vars)
    save_path = "tmp/%s.ckpt" % self.name
    saver.restore(sess, save_path)
    print("Model restored from file: %s" % save_path)


class MLP(Model):
  def __init__(self, placeholders, input_dim, **kwargs):
    super(MLP, self).__init__(**kwargs)

    self.inputs = placeholders['features']
    self.input_dim = input_dim
    # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
    self.output_dim = placeholders['labels'].get_shape().as_list()[1]
    self.placeholders = placeholders

    self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    self.build()

  def _loss(self):
    # Weight decay loss
    for var in self.layers[0].vars.values():
      self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

    # Cross entropy error
    self.loss += my_softmax_cross_entropy(self.outputs, self.placeholders['labels'])

  def _loss_reg(self):
    # Weight decay loss
    for var in self.layers[0].vars.values():
      self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

    # regression loss
    self.loss += tf.reduce_mean(tf.square(self.outputs - self.placeholders['labels']))

  def _accuracy(self):
    self.accuracy = my_accuracy(self.outputs, self.placeholders['labels'])

  def _build(self):
    self.layers.append(Dense(input_dim=self.input_dim,
                             output_dim=FLAGS.hidden1,
                             placeholders=self.placeholders,
                             act=tf.nn.relu,
                             dropout=True,
                             sparse_inputs=True,
                             logging=self.logging))

    self.layers.append(Dense(input_dim=FLAGS.hidden1,
                             output_dim=self.output_dim,
                             placeholders=self.placeholders,
                             act=lambda x: x,
                             skip_connection=False,
                             dropout=True,
                             logging=self.logging))

  def predict(self):
    return tf.nn.softmax(self.outputs)


class GCN_DEEP_DIVER(Model):
  def __init__(self, placeholders, input_dim, **kwargs):
    super(GCN_DEEP_DIVER, self).__init__(**kwargs)

    self.inputs = placeholders['features']
    self.input_dim = input_dim
    # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
    self.output_dim = placeholders['labels'].get_shape().as_list()[1]
    self.placeholders = placeholders

    self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    self.build()

  def _loss(self):
    # Weight decay loss
    for var in self.layers[0].vars.values():
      self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

    # 32 outputs
    diver_loss = my_softmax_cross_entropy(self.outputs[:, 0:self.output_dim], self.placeholders['labels'])
    for i in range(1, FLAGS.diver_num):
      diver_loss = tf.reduce_min([diver_loss, my_softmax_cross_entropy(self.outputs[:, 2 * i:2 * i + self.output_dim],
                                                                       self.placeholders['labels'])])
    self.loss += diver_loss

  def _loss_reg(self):
    # Weight decay loss
    for var in self.layers[0].vars.values():
      self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

    # regression loss
    self.loss += tf.reduce_mean(tf.abs(self.outputs - self.placeholders['labels']))

  def _accuracy(self):
    # 32 outputs
    acc = my_accuracy(self.outputs[:, 0:self.output_dim], self.placeholders['labels'])
    for i in range(1, FLAGS.diver_num):
      acc = tf.reduce_max(
          [acc, my_accuracy(self.outputs[:, 2 * i:2 * i + self.output_dim], self.placeholders['labels'])])
    self.accuracy = acc

  def _build(self):

    _LAYER_UIDS['graphconvolution'] = 0
    self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                        output_dim=FLAGS.hidden1,
                                        placeholders=self.placeholders,
                                        act=tf.nn.relu,
                                        dropout=True,
                                        sparse_inputs=True,
                                        logging=self.logging))
    for i in range(FLAGS.num_layer - 2):
      self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden1,
                                          placeholders=self.placeholders,
                                          act=tf.nn.relu,
                                          dropout=True,
                                          logging=self.logging))
    self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                        output_dim=2 * FLAGS.diver_num,
                                        placeholders=self.placeholders,
                                        skip_connection=False,
                                        act=lambda x: x,
                                        dropout=True,
                                        logging=self.logging))

  def predict(self):
    return tf.nn.softmax(self.outputs)


class GCN_DEEP_DIVER_DIMES(Model):
  def __init__(self, placeholders, input_dim, **kwargs):
    super(GCN_DEEP_DIVER_DIMES, self).__init__(**kwargs)

    self.inputs = placeholders['features']
    self.input_dim = input_dim
    # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
    self.output_dim = placeholders['labels'].get_shape().as_list()[1]
    self.placeholders = placeholders
    self.weights = []

    self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    self.meta_update = kwargs.get('meta_update')
    self.meta_steps = kwargs.get('meta_steps')
    self.aux_grad = kwargs.get('aux_grad', False)
    self.use_ppo = kwargs.get('ppo')
    self.random_heatmap = kwargs.get('random_heatmap')
    self.normalize_factor = kwargs.get('normalize_factor')
    self.supervision_factor = kwargs.get('supervision_factor')
    self.inner_optimizer = None
    self.reset_inner_op = None
    self.output_std = None
    self.summary = {}

    self.build()

  def _loss(self):
    if self.supervision_factor != 0 and self.supervision_factor is not None:
      print("supervision_factor is", self.supervision_factor)
      # Weight decay loss
      for var in self.layers[0].vars.values():
        self.loss += self.supervision_factor * FLAGS.weight_decay * tf.nn.l2_loss(var)
      # 32 outputs
      diver_loss = my_softmax_cross_entropy(
          self.outputs[:, 0:self.output_dim], self.placeholders['labels'])
      for i in range(1, FLAGS.diver_num):
        diver_loss = tf.reduce_min([diver_loss, my_softmax_cross_entropy(
            self.outputs[:, 2 * i:2 * i + self.output_dim], self.placeholders['labels'])])
      self.loss += self.supervision_factor * diver_loss
      self.summary['supervised_loss'] = diver_loss

    outputs = tf.nn.log_softmax(tf.reshape(self.outputs, [-1, FLAGS.diver_num, 2]), axis=-1)
    outputs = outputs[:, :, 1] - outputs[:, :, 0]
    if self.output_std is None:
      output_std = tf.math.maximum(1.0, self.normalize_factor *
                                   tf.math.reduce_std(outputs, axis=0, keepdims=True))
      self.output_std = output_std
    else:
      output_std = self.output_std
    # output_std = tf.math.maximum(1.0, self.normalize_factor *
    #                              tf.math.reduce_std(outputs, axis=0, keepdims=True))
    outputs = outputs / output_std
    self.outputs_softmax = tf.reshape(outputs, [-1, FLAGS.diver_num, 1])
    self.outputs_softmax = tf.concat([tf.zeros_like(self.outputs_softmax), self.outputs_softmax], axis=-1)
    self.outputs_softmax = tf.reshape(self.outputs_softmax, [-1, FLAGS.diver_num * 2])

    tf.summary.histogram('outputs', outputs)
    acc = tf_decode(self.placeholders['support'][1].indices, outputs,
                    FLAGS.diver_num,
                    self.placeholders['max_graph_size'],
                    self.placeholders['max_num_edges'])
    self.accuracy = tf.reduce_max(acc)
    self.summary['accuracy'] = self.accuracy

    # best_idx = tf.math.top_k(acc, k=4).indices
    # outputs = tf.transpose(outputs, [1, 0])
    # with tf.control_dependencies([tf.print(tf.gather(acc, best_idx)),
    #                               tf.print(tf.reduce_max(outputs), tf.reduce_min(outputs))]):
    #     outputs = tf.gather(outputs, best_idx)
    # outputs = tf.transpose(outputs, [1, 0])
    # diver_loss = tf_par_grad(self.placeholders['support'][1].indices, outputs, 4)
    # diver_loss = - tf.reduce_sum(diver_loss * outputs)

    if self.use_ppo:
      diver_loss, old_log_prob, jax_path, old_reward = tf_par_grad_ppo(
          self.placeholders['support'][1].indices,
          outputs,
          self.placeholders['old_log_prob'],
          self.placeholders['jax_path'],
          self.placeholders['old_reward'],
          FLAGS.diver_num,
          self.placeholders['max_graph_size'],
          self.placeholders['max_num_edges'])
      self.old_log_prob = old_log_prob
      self.jax_path = jax_path
      self.old_reward = old_reward
    else:
      diver_loss, old_reward, stat_abs_mean, stat_std = tf_par_grad(self.placeholders['support'][1].indices,
                                                                    outputs, FLAGS.diver_num,
                                                                    self.placeholders['max_graph_size'],
                                                                    self.placeholders['max_num_edges'],
                                                                    self.aux_grad)
      self.summary['stats/abs_mean'] = stat_abs_mean
      self.summary['stats/std_dev'] = stat_std
      self.old_log_prob = None
      self.jax_path = None
      self.old_reward = None
    diver_loss = - tf.reduce_sum(diver_loss * outputs)

    self.loss += diver_loss
    self.summary['loss'] = diver_loss
    self.summary['accuracy_sample'] = tf.reduce_max(old_reward)

    div_reward = tf.reshape(outputs, [-1, FLAGS.diver_num, 1]) - tf.reshape(outputs, [-1, 1, FLAGS.diver_num])
    div_reward = div_reward - tf.eye(FLAGS.diver_num) * div_reward
    div_reward = tf.reduce_mean(tf.math.abs(div_reward), axis=0)
    div_reward = div_reward + tf.eye(FLAGS.diver_num) * 1e9
    div_reward = tf.reduce_min(div_reward, axis=1)
    div_reward = tf.reduce_mean(div_reward)
    if self.use_ppo:
      div_weight = 0.00001
    else:
      div_weight = 0.0001 * tf.maximum(1.0, tf.math.abs(diver_loss) / 10.0)
    self.loss -= div_weight * div_reward
    self.summary['div_reward'] = div_reward

  def _loss_reg(self):
    # Weight decay loss
    # for var in self.layers[0].vars.values():
    #     self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
    # regression loss
    # self.loss += tf.reduce_mean(tf.abs(self.outputs-self.placeholders['labels']))
    pass

  def _accuracy(self):
    # 32 outputs
    # outputs = tf.nn.log_softmax(tf.reshape(self.outputs, [-1, FLAGS.diver_num, 2]), axis=-1)
    # outputs = outputs[:, :, 1] - outputs[:, :, 0]
    # acc = tf_decode(self.placeholders['support'][1].indices, outputs, FLAGS.diver_num)
    # self.accuracy = tf.reduce_max(acc)
    pass

  def _build(self):

    _LAYER_UIDS['graphconvolution'] = 0
    self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                        output_dim=FLAGS.hidden1,
                                        placeholders=self.placeholders,
                                        act=tf.nn.relu,
                                        dropout=True,
                                        sparse_inputs=True,
                                        logging=self.logging))
    self.weights.append(self.layers[-1].vars)

    for i in range(FLAGS.num_layer - 2):
      self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden1,
                                          placeholders=self.placeholders,
                                          act=tf.nn.relu,
                                          dropout=True,
                                          logging=self.logging))
      self.weights.append(self.layers[-1].vars)

    self.layers.append(Dense(input_dim=FLAGS.hidden1,
                             output_dim=FLAGS.hidden1,
                             placeholders=self.placeholders,
                             act=tf.nn.relu,
                             repeat=10,
                             dropout=True,
                             logging=self.logging))
    self.weights.append(self.layers[-1].vars)

    self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                        output_dim=2 * FLAGS.diver_num,
                                        placeholders=self.placeholders,
                                        act=lambda x: x,
                                        dropout=True,
                                        mis_bias=False,
                                        logging=self.logging))
    self.weights.append(self.layers[-1].vars)

  def build(self):
    """ Wrapper for _build() """
    with tf.variable_scope(self.name):
      self._build()

    flatten_list = lambda regular_list: [item for sublist in regular_list for item in sublist]

    self.forward(self.weights)
    if self.meta_update:
      print("Using meta update")
      updated_weights = 2
      fast_weights = self.weights[-updated_weights:]
      grad_1st_moment = dict(flatten_list([
          [(f'l{l}' + key, tf.zeros_like(weight[key])) for key in weight.keys()]
          for l, weight in enumerate(fast_weights)]))
      grad_2nd_moment = dict(flatten_list([
          [(f'l{l}' + key, tf.zeros_like(weight[key])) for key in weight.keys()]
          for l, weight in enumerate(fast_weights)]))
      meta_learning_rate = 0.0002
      beta1 = 0.9
      beta2 = 0.98
      epsilon = 1e-6
      for i in range(self.meta_steps):
        grads = tf.gradients(self.loss, flatten_list([weight.values() for weight in fast_weights]))
        grads = [tf.stop_gradient(grad) for grad in grads]
        var_names = flatten_list([[f'l{l}' + key for key in weight.keys()] for l, weight in enumerate(fast_weights)])
        gvs = dict(zip(var_names, grads))
        grad_1st_moment = dict(
            (key, grad_1st_moment[key] * beta1 + gvs[key] * (1 - beta1))
            for key in grad_1st_moment.keys())
        grad_2nd_moment = dict(
            (key, grad_2nd_moment[key] * beta2 + tf.math.square(gvs[key]) * (1 - beta2))
            for key in grad_2nd_moment.keys())
        new_fast_weights = []
        for l, weight in enumerate(fast_weights):
          new_fast_weights.append(dict(zip(weight.keys(), [
              weight[key]
              - meta_learning_rate * (
                  (grad_1st_moment[f'l{l}' + key] / (1 - beta1 ** (i + 1))) /
                  (tf.math.sqrt(grad_2nd_moment[f'l{l}' + key] / (1 - beta2 ** (i + 1))) + epsilon))
              for key in weight.keys()
          ])))
        fast_weights = new_fast_weights
        self.loss = 0.0
        self.forward(self.weights[:-updated_weights] + fast_weights)
    self._accuracy()
    self.opt_op = self.optimizer.minimize(self.loss)

    trainable_vars = tf.trainable_variables()
    accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in trainable_vars]
    self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
    grads_and_vars = self.optimizer.compute_gradients(self.loss)
    # self.opt_op = self.optimizer.apply_gradients(grads_and_vars)
    self.accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(grads_and_vars)]
    clipped_accum_vers, _ = tf.clip_by_global_norm(accum_vars, 0.5)
    self.train_ops = self.optimizer.apply_gradients(
        [(clipped_accum_vers[i], gv[1]) for i, gv in enumerate(grads_and_vars)])

  def forward(self, weights):
    # Build sequential layer model
    layer_id = 0
    self.activations.append(self.inputs)
    for layer, weight in zip(self.layers, weights):
      if self.name == 'gcn_deep' and layer_id % 2 == 0 and 0 < layer_id < len(self.layers) - 1:
        hidden = layer(self.activations[-1], weight)
        self.activations.append(tf.nn.relu(hidden + self.activations[-2]))
        layer_id = layer_id + 1
      # elif layer_id < len(self.layers)-1:
      #     hidden = tf.nn.relu(layer(self.activations[-1], weight))
      #     self.activations.append(hidden)
      #     layer_id = layer_id + 1
      else:
        hidden = layer(self.activations[-1], weight)
        self.activations.append(hidden)
        layer_id = layer_id + 1
    self.outputs = self.activations[-1]

    if self.random_heatmap:
      print("Using random heatmap")
      self.outputs = self.outputs * 1e-9 + tf.random.uniform(tf.shape(self.outputs))

    # if self.name != 'gcn_dqn':
    #   self.outputs_softmax = tf.nn.log_softmax(self.outputs[:, 0:2])
    # if self.name == 'gcn_deep_diver' or 'gcn_deep_diver_dimes':
    #   for out_id in range(1, FLAGS.diver_num):
    #     self.outputs_softmax = tf.concat([self.outputs_softmax, tf.nn.log_softmax(
    #         self.outputs[:, self.output_dim * out_id:self.output_dim * (out_id + 1)])], axis=1)

    if self.name == 'gcn_dqn':
      self.pred = tf.argmax(self.outputs)
    # Store model variables for easy access
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
    self.vars = {var.name: var for var in variables}

    # Build metrics
    self._loss()

  def predict(self):
    return tf.nn.softmax(self.outputs)


@tf.function(input_signature=[tf.TensorSpec(None, tf.int64), tf.TensorSpec(None, tf.float32),
                              tf.TensorSpec(None, tf.int64), tf.TensorSpec(None, tf.int64),
                              tf.TensorSpec(None, tf.int64), tf.TensorSpec(None, tf.bool)])
def tf_par_grad(adj, par, diver_num, max_graph_size, max_num_edges, aux_grad):
  y, z, st1, st2 = tf.numpy_function(reinforce.np_get_merged_grad,
                                     [adj, par, diver_num, max_graph_size, max_num_edges, aux_grad],
                                     (tf.float32, tf.float32, tf.float32, tf.float32))
  return y, z, st1, st2


@tf.function(input_signature=[tf.TensorSpec(None, tf.int64), tf.TensorSpec(None, tf.float32),
                              tf.TensorSpec(None, tf.float32), tf.TensorSpec(None, tf.int64),
                              tf.TensorSpec(None, tf.float32),
                              tf.TensorSpec(None, tf.int64), tf.TensorSpec(None, tf.int64),
                              tf.TensorSpec(None, tf.int64)])
def tf_par_grad_ppo(adj, par, old_log_prob, jax_path, old_reward, diver_num, max_graph_size, max_num_edges):
  x, y, z, w = tf.numpy_function(reinforce.np_get_merged_grad_ppo,
                                 [adj, par, old_log_prob, jax_path, old_reward,
                                  diver_num, max_graph_size, max_num_edges],
                                 (tf.float32, tf.float32, tf.int64, tf.float32))
  return x, y, z, w


@tf.function(input_signature=[tf.TensorSpec(None, tf.int64), tf.TensorSpec(None, tf.float32),
                              tf.TensorSpec(None, tf.int64), tf.TensorSpec(None, tf.int64),
                              tf.TensorSpec(None, tf.int64)])
def tf_decode(adj, par, diver_num, max_graph_size, max_num_edges):
  y = tf.numpy_function(reinforce.np_get_mis, [adj, par, diver_num, max_graph_size, max_num_edges], tf.float32)
  return y
