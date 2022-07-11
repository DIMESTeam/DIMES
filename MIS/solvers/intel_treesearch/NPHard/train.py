from __future__ import division
from __future__ import print_function

import sys
import os
import random
sys.path.append( '%s/gcn' % os.path.dirname(os.path.realpath(__file__)) )

import time
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
from copy import deepcopy

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from utils import *
from models import GCN_DEEP_DIVER
from pathlib import Path

### Begin argument parsing
import argparse

parser = argparse.ArgumentParser(description="Intel-based tree search.")
parser.add_argument("input", type=str, action="store", help="Directory containing input graphs (to be solved/trained on).")
parser.add_argument("output", type=str, action="store",  help="Folder in which the output (e.g. json containg statistics and solution will be stored, or trained weights)")

parser.add_argument("--cuda_device", type=int, nargs="*", action="store", default=0, help="Which cuda device should be used")
parser.add_argument("--self_loops", action="store_true", default=False, help="Enable self loops addition (in input data) for GCN-based model.")
parser.add_argument("--model_prob_maps", type=int, action="store", default=32, help="Number of probability maps the model was/should be trained for.")
parser.add_argument("--lr", type=float, action="store", default=0.001, help="Learning rate (for training)")
parser.add_argument("--epochs", type=int, action="store", default=20, help="Number of epochs to train for (notion changed compared to original Intel version, see paper for details)")
parser.add_argument("--pretrained_weights", type=str, action="store", help="Pre-trained weights to continue training on")

args = parser.parse_args()


N_bd = 32

# Settings
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', args.lr, 'Initial learning rate.')
flags.DEFINE_integer('epochs', args.epochs, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('diver_num', args.model_prob_maps, 'Number of outputs.')
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probaNUmbility).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 1000, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 1, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('num_layer', 20, 'number of layers.')

# we need to define our argparse argument here aswell, otherwise tf.flags throws an exception
flags.DEFINE_string("cuda_device", "", "")
flags.DEFINE_boolean("self_loops", False, "")
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
model_func = GCN_DEEP_DIVER

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=(None, N_bd)), # featureless: #points
    'labels': tf.placeholder(tf.float32, shape=(None, 2)), # 0: not linked, 1:linked
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=N_bd, logging=True)

# use gpu 0
os.environ['CUDA_VISIBLE_DEVICES']=str(args.cuda_device)

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
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

ckpt = None
if args.pretrained_weights:
    ckpt=tf.train.get_checkpoint_state(args.pretrained_weights)
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

# cost_val = []

all_loss = np.zeros(len(train_mat_names), dtype=float)
all_acc = np.zeros(len(train_mat_names), dtype=float)

if args.self_loops:
    import scipy

# Train model
for epoch in range(FLAGS.epochs):
    if os.path.isdir(args.output + "/%04d"%epoch):
        continue
    ct = 0
    os.makedirs(args.output+"/%04d" % epoch)
    # for id in np.random.permutation(len(train_mat_names)):
    ids = list(range(len(train_mat_names)))
    random.shuffle(ids)
    ct_all = len(ids)
    for id in ids:
        ct = ct + 1
        t = time.time()
        # load data
        mat_contents = sio.loadmat(data_path+'/'+train_mat_names[id])
        adj = mat_contents['adj']

        if args.self_loops:
            identity = scipy.sparse.identity(adj.shape[0], dtype=adj.dtype, format=adj.format)
            adj = adj + identity

        if 'indset_label' not in mat_contents:
            continue
        yy = mat_contents['indset_label']
        nn, nr = yy.shape # number of nodes & results
        # y_train = yy[:,np.random.randint(0,nr)]
        # y_train = np.concatenate([1-np.expand_dims(y_train,axis=1), np.expand_dims(y_train,axis=1)],axis=1)

        # sample an intermediate graph
        yyr = yy[:, np.random.randint(0, nr)]
        yyr_num = np.sum(yyr)
        if yyr_num == 0:
            continue
        yyr_down_num = np.random.randint(0, yyr_num)
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

        y_train = np.concatenate([1 - np.expand_dims(yyr, axis=1), np.expand_dims(yyr, axis=1)], axis=1)

        features = np.ones([nn, N_bd])
        features = sp.lil_matrix(features)
        features = preprocess_features(features)
        support = simple_polynomials(adj, FLAGS.max_degree)
        # print(features)
        train_mask = np.ones([nn, 1], dtype=bool)

        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs], feed_dict=feed_dict)
        all_loss[ct-1] = outs[1]
        all_acc[ct-1] = outs[2]

        # Print results
        print('%03d %05d/%05d' % (epoch + 1, ct, ct_all),
              "train_loss=", "{:.5f}".format(np.mean(all_loss[np.where(all_loss)])),
              "train_acc=", "{:.5f}".format(np.mean(all_acc[np.where(all_acc)])),
              "time=", "{:.5f}".format(time.time() - t))


    target=open(args.output+"/%04d/score.txt"%epoch,'w')
    target.write("%f\n%f\n"%(np.mean(all_loss[np.where(all_loss)]),np.mean(all_acc[np.where(all_acc)])))
    target.close()

    saver.save(sess,args.output + "/model.ckpt")
    saver.save(sess,args.output + "/%04d/model.ckpt"%epoch)

print("Optimization Finished!")
