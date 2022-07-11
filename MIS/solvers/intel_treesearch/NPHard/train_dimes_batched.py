from __future__ import division
from __future__ import print_function

import sys
import os
import random

sys.path.append('%s/gcn' % os.path.dirname(os.path.realpath(__file__)))

import time
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
from copy import deepcopy

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from utils import *
import reinforce
from models import GCN_DEEP_DIVER_DIMES
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# Begin argument parsing
import argparse

parser = argparse.ArgumentParser(description="Intel-based tree search.")
parser.add_argument("input", type=str, action="store",
                    help="Directory containing input graphs (to be solved/trained on).")
parser.add_argument("output", type=str, action="store",
                    help="Folder in which the output (e.g. json containg statistics and solution will be stored, or trained weights)")

parser.add_argument("--cuda_device", type=int, nargs="*", action="store", default=0,
                    help="Which cuda device should be used")
parser.add_argument("--meta_update", action="store_true", default=False, help="Enable meta update.")
parser.add_argument("--meta_steps", type=int, action="store", default=1, help="Number of meta steps.")
parser.add_argument("--aux_grad", action="store_true", default=False, help="Enable aux grad.")
parser.add_argument("--batch_size", type=int, action="store", default=8, help="Batch size.")
parser.add_argument("--PPO_inner_steps", type=int, action="store", default=4, help="Number of inner PPO steps.")
parser.add_argument("--normalize_factor", type=float, action="store", default=2.0, help="Normalize factor.")
parser.add_argument("--supervision_factor", type=float, action="store", default=0.0, help="Supervision factor.")
parser.add_argument("--ppo", action="store_true", default=False, help="Enable PPO.")
parser.add_argument("--random_prune", action="store_true", default=False, help="Enable random pruning.")
parser.add_argument("--self_loops", action="store_true", default=False,
                    help="Enable self loops addition (in input data) for GCN-based model.")
parser.add_argument("--model_prob_maps", type=int, action="store", default=32,
                    help="Number of probability maps the model was/should be trained for.")
parser.add_argument("--lr", type=float, action="store", default=0.001, help="Learning rate (for training)")
parser.add_argument("--epochs", type=int, action="store", default=200,
                    help="Number of epochs to train for (notion changed compared to original Intel version, see paper for details)")
parser.add_argument("--pretrained_weights", type=str, action="store",
                    help="Pre-trained weights to continue training on")

args = parser.parse_args()

N_bd = 32

# Settings
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', args.lr, 'Initial learning rate.')
flags.DEFINE_integer('epochs', args.epochs, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('diver_num', args.model_prob_maps, 'Number of outputs.')
flags.DEFINE_integer('batch_size', args.batch_size, 'Minibatch size.')
flags.DEFINE_integer('ppo_inner_steps', args.PPO_inner_steps, 'Number of inner PPO steps.')
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probaNUmbility).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 1000, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 1, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('num_layer', 20, 'number of layers.')

# we need to define our argparse argument here aswell, otherwise tf.flags throws an exception
flags.DEFINE_string("cuda_device", "", "")
flags.DEFINE_boolean("self_loops", False, "")
flags.DEFINE_boolean("meta_update", False, "")
flags.DEFINE_integer("meta_steps", 1, "")
flags.DEFINE_boolean("aux_grad", False, "")
flags.DEFINE_float("normalize_factor", 2.0, "")
flags.DEFINE_float("supervision_factor", 0.0, "")
flags.DEFINE_boolean("ppo", False, "")
flags.DEFINE_boolean("random_prune", False, "")
flags.DEFINE_boolean("reduction", False, "")
flags.DEFINE_boolean("local_search", False, "")
flags.DEFINE_string("model_prob_maps", "", "")
flags.DEFINE_string("lr", "", "")
flags.DEFINE_string("pretrained_weights", "", "")

# Load data
data_path = args.input
if not Path(data_path).exists():
  raise ValueError(f"Input directory {data_path} does not exists")
train_mat_names = os.listdir(data_path)

# Some preprocessing
num_supports = 1 + FLAGS.max_degree
model_func = GCN_DEEP_DIVER_DIMES

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# use gpu 0
# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device[0])
# print(os.environ['CUDA_VISIBLE_DEVICES'])

reinforce.beam_size = 1024
# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=(None, N_bd)),  # featureless: #points
    'labels': tf.placeholder(tf.float32, shape=(None, 2)),  # 0: not linked, 1:linked
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'old_reward': tf.placeholder(tf.float32, shape=(FLAGS.diver_num, reinforce.beam_size)),
    'old_log_prob': tf.placeholder(tf.float32, shape=(FLAGS.diver_num, reinforce.beam_size, None)),
    'jax_path': tf.placeholder(tf.int64, shape=(FLAGS.diver_num, reinforce.beam_size, None)),
    'max_graph_size': tf.placeholder(tf.int64, shape=()),
    'max_num_edges': tf.placeholder(tf.int64, shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=N_bd, logging=True, normalize_factor=FLAGS.normalize_factor,
                   meta_update=FLAGS.meta_update, meta_steps=FLAGS.meta_steps, ppo=FLAGS.ppo,
                   supervision_factor=FLAGS.supervision_factor, aux_grad=FLAGS.aux_grad)

# Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
  t_test = time.time()
  feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
  outs_val = sess.run([model.loss, model.accuracy, model.outputs_softmax], feed_dict=feed_dict_val)
  return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2]


# Init variables
saver = tf.train.Checkpoint(optimizer=model.optimizer, variables=tf.trainable_variables())
manager = tf.train.CheckpointManager(
    saver, directory=args.output, max_to_keep=2)
sess.run(tf.global_variables_initializer())

ckpt = None
if args.pretrained_weights:
  ckpt = tf.train.latest_checkpoint(args.pretrained_weights)
  if ckpt:
    print(f"Restoring pretrained weights from {args.pretrained_weights}: {ckpt}")
    status = saver.restore(ckpt)
    status.run_restore_ops(session=sess)
  else:
    print(f"No pretrained weights found at {args.pretrained_weights}: {ckpt}")
else:
  print("No pretrained weights provided")
with sess.as_default():
  manager.save()

# cost_val = []

if args.self_loops:
  import scipy

import datetime

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = args.output + '/logs/train'
print(train_log_dir)
writer = tf.summary.FileWriter(train_log_dir)
torch_writer = SummaryWriter(train_log_dir + '_scalar')
summaries = tf.compat.v1.summary.merge_all()

max_graph_size = 0
max_num_edges = 0
step = 0
# Train model

mean_summary = {}
for epoch in range(FLAGS.epochs):
  if "large" in args.output:
    print("Using large dataset")
    epoch_size = 50
  else:
    epoch_size = 1000
  if os.path.isdir(args.output + "/%04d" % epoch):
    print("Skipping epoch %d since output directory already exists" % epoch)
    step += epoch_size
    continue
  ct = 0
  all_loss = np.zeros(len(train_mat_names), dtype=float)
  all_acc = np.zeros(len(train_mat_names), dtype=float)
  all_acc_gt = np.zeros(len(train_mat_names), dtype=float)
  # for id in np.random.permutation(len(train_mat_names)):
  ids = list(range(len(train_mat_names)))
  random.Random(epoch + 42).shuffle(ids)
  ct_all = len(ids)
  ids = ids[:epoch_size]

  batches = []
  for id in ids:
    ct = ct + 1
    t = time.time()
    # load data
    mat_contents = sio.loadmat(data_path + '/' + train_mat_names[id])
    adj = mat_contents['adj']
    if args.self_loops:
      identity = scipy.sparse.identity(adj.shape[0], dtype=adj.dtype, format=adj.format)
      adj = adj + identity

    if 'indset_label' not in mat_contents:
      yyr = np.zeros(adj.shape[0], dtype=int)
      yyr[0] = 1
      nn = yyr.shape[0]
    else:
      yy = mat_contents['indset_label']
      nn, nr = yy.shape  # number of nodes & results
      yyr = yy[:, np.random.randint(0, nr)]
      if FLAGS.random_prune:
        # y_train = yy[:, np.random.randint(0, nr)]
        # y_train = np.concatenate([1-np.expand_dims(y_train, axis=1), np.expand_dims(y_train,axis=1)],axis=1)

        # sample an intermediate graph
        yyr_num = np.sum(yyr)
        yyr_down_num = np.random.randint(0, max(int(yyr_num * 0.8), 0))
        if yyr_down_num > 0:
            yyr_down_prob = yyr * np.random.random_sample(yyr.shape)
            yyr_down_flag = (yyr_down_prob >= np.partition(yyr_down_prob, -yyr_down_num)[-yyr_down_num])
            tmp = np.sum(adj[yyr_down_flag, :], axis=0) > 0
            tmp = np.asarray(tmp).reshape(-1)
            yyr_down_flag[tmp] = 1
            adj_down = adj[yyr_down_flag == 0, :]
            adj_down = adj_down[:, yyr_down_flag == 0]
            yyr_down = yyr[yyr_down_flag == 0]
            adj = adj_down
            nn = yyr_down.shape[0]
            yyr = yyr_down

    if int(nn) > max_graph_size:
      max_graph_size = int(nn * 1.2)
    if int(adj.count_nonzero()) > max_num_edges:
      max_num_edges = int(adj.count_nonzero() * 1.2)
    y_train = np.concatenate([1 - np.expand_dims(yyr, axis=1), np.expand_dims(yyr, axis=1)], axis=1)

    features = np.ones([nn, N_bd])
    features = sp.lil_matrix(features)
    features = preprocess_features(features)
    support = simple_polynomials(adj, FLAGS.max_degree)

    train_mask = np.ones([nn, 1], dtype=bool)

    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({placeholders['max_graph_size']: max_graph_size})
    feed_dict.update({placeholders['max_num_edges']: max_num_edges})
    old_reward = np.zeros((FLAGS.diver_num, reinforce.beam_size))
    old_log_prob = np.ones((FLAGS.diver_num, reinforce.beam_size, max_graph_size))
    jax_path = np.zeros((FLAGS.diver_num, reinforce.beam_size, max_graph_size), dtype=np.int64)

    # print(adj.count_nonzero())
    # print(np.allclose(adj.todense(), adj.todense().T))
    # print(feed_dict[placeholders['support'][0]][0].shape)
    # print(feed_dict[placeholders['support'][1]][0].shape)
    # raise NotImplemented

    batches.append((feed_dict, old_reward, old_log_prob, jax_path, None))

    if len(batches) == FLAGS.batch_size:
      # Training step
      for inner_step in range(FLAGS.PPO_inner_steps if FLAGS.ppo else 1):
        feed_dict, old_reward, old_log_prob, jax_path, obj = batches[0]
        feed_dict.update({placeholders['old_reward']: old_reward})
        feed_dict.update({placeholders['old_log_prob']: old_log_prob})
        feed_dict.update({placeholders['jax_path']: jax_path})
        sess.run([model.zero_ops], feed_dict=feed_dict)
        for i in range(len(batches)):
          feed_dict, old_reward, old_log_prob, jax_path, obj = batches[i]
          feed_dict.update({placeholders['old_reward']: old_reward})
          feed_dict.update({placeholders['old_log_prob']: old_log_prob})
          feed_dict.update({placeholders['jax_path']: jax_path})
          if FLAGS.ppo:
            outs = sess.run([summaries, model.accum_ops, model.loss, model.accuracy, model.outputs,
                             model.old_log_prob, model.jax_path, model.old_reward, model.summary],
                            feed_dict=feed_dict)
          else:
            outs = sess.run([summaries, model.accum_ops, model.loss, model.accuracy, model.outputs,
                             model.summary],
                            feed_dict=feed_dict)
          summ = outs[0]
          all_loss[ct - 1] = outs[2]
          all_acc[ct - 1] = outs[3]
          all_acc_gt[ct - 1] = np.sum(yyr)

          for key in outs[-1]:
            if key not in mean_summary:
              mean_summary[key] = []
            mean_summary[key].append(outs[-1][key])

          if len(mean_summary[list(mean_summary.keys())[0]]) == max(round(epoch_size / 16), 4):
            for key in mean_summary:
              results = np.mean(mean_summary[key], axis=0)
              torch_writer.add_scalar(key, results, step)
            torch_writer.flush()
            mean_summary = {}

          if FLAGS.ppo:
            old_log_prob = outs[5]
            jax_path = outs[6]
            old_reward = outs[7]
          batches[i] = (feed_dict, old_reward, old_log_prob, jax_path, obj)
          writer.add_summary(summ, global_step=step)
          step += 1
        sess.run([model.train_ops], feed_dict=feed_dict)
      batches = []

  os.makedirs(args.output + "/%04d" % epoch)
  target = open(args.output + "/%04d/score.txt" % epoch, 'w')
  target.write(f"{np.mean(all_loss[np.where(all_loss)]):f}\n{np.mean(all_acc[np.where(all_acc)]):f}\n")
  target.close()

  saver.save(args.output + "/%04d/ckpt" % epoch, session=sess)
  with sess.as_default():
    manager.save()

print("Optimization Finished!")
