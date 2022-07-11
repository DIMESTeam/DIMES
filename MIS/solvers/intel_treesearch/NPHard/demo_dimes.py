from __future__ import division
from __future__ import print_function

### Begin argument parsing
import argparse

parser = argparse.ArgumentParser(description="Intel-based tree search.")
parser.add_argument("input", type=str, action="store", help="Directory containing input graphs to be solved")
parser.add_argument("output", type=str, action="store",  help="Folder in which the output will be stored")
parser.add_argument("pretrained_weights", type=str, action="store", help="Pre-trained weights to be used for solving (folder containg checkpoints)")

parser.add_argument("--time_limit", type=int, nargs="?", action="store", default=600, help="Time limit in seconds")
parser.add_argument("--cuda_device", type=int, nargs="*", action="store", default=0, help="Which cuda device should be used")
parser.add_argument("--self_loops", action="store_true", default=False, help="Enable self loops addition (in input data) for GCN-based model.")
parser.add_argument("--dimes", action="store_true", default=False, help="Enable DIMES-based model.")
parser.add_argument("--meta_update", action="store_true", default=False, help="Enable meta-update.")
parser.add_argument("--meta_steps", type=int, action="store", default=1, help="Number of meta-update steps.")
parser.add_argument("--hidden1", type=int, action="store", default=64, help="Number of hidden units in first hidden layer.")
parser.add_argument("--normalize_factor", type=float, default=2.0, help="Normalization factor for DIMES-based model.")
parser.add_argument("--reduction", action="store_true", default=False, help="Enable reduction of graph (kernelization).")
parser.add_argument("--random_heatmap", action="store_true", default=False, help="use random_heatmap.")
parser.add_argument("--local_search", action="store_true", default=False, help="Enable local search if time left.")
parser.add_argument("--model_prob_maps", type=int, action="store", default=32, help="Number of probability maps.")

args = parser.parse_args()

prob_maps = args.model_prob_maps

### End argument parsing

import sys
import os
sys.path.append( '%s/gcn' % os.path.dirname(os.path.realpath(__file__)) )
# add the libary path for graph reduction and local search

if args.reduction or args.local_search:
    sys.path.append( '%s/kernel' % os.path.dirname(os.path.realpath(__file__)) )

import time
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
from copy import deepcopy

# import the libary for graph reduction and local search
if args.reduction or args.local_search:
    from reduce_lib import reducelib

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.config.experimental.set_visible_devices([], "GPU")
from utils import *
from models import GCN_DEEP_DIVER_DIMES, GCN_DEEP_DIVER
import reinforce

import statistics

N_bd = 32

# Settings
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 201, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('diver_num', 32, 'Number of outputs.')
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probaNUmbility).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 1000, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 1, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('num_layer', 20, 'number of layers.')
# we need to define our argparse argument here aswell, otherwise tf.flags throws an exception
flags.DEFINE_string("time_limit", "", "")
flags.DEFINE_string("cuda_device", "", "")
flags.DEFINE_boolean("self_loops", False, "")
flags.DEFINE_boolean("dimes", False, "")
flags.DEFINE_float("normalize_factor", 2.0, "")
flags.DEFINE_boolean("meta_update", False, "")
flags.DEFINE_integer("meta_steps", 1, "")
flags.DEFINE_boolean("reduction", False, "")
flags.DEFINE_boolean("random_heatmap", False, "")
flags.DEFINE_boolean("local_search", False, "")
flags.DEFINE_string("model_prob_maps", "", "")

# test data path
data_path = args.input
val_mat_names = os.listdir(data_path)

# Some preprocessing

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# use gpu 0
# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device[0])
# print(os.environ['CUDA_VISIBLE_DEVICES'])


num_supports = 1 + FLAGS.max_degree
if args.dimes:
    model_func = GCN_DEEP_DIVER_DIMES
else:
    model_func = GCN_DEEP_DIVER

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=(None, N_bd)),  # featureless: #points
    'labels': tf.placeholder(tf.float32, shape=(None, 2)),  # 0: not linked, 1:linked
    'labels_mask': tf.placeholder(tf.int32),
    'max_graph_size': tf.placeholder(tf.int64),
    'max_num_edges': tf.placeholder(tf.int64),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=N_bd, logging=True, normalize_factor=args.normalize_factor,
                   random_heatmap=args.random_heatmap, meta_update=args.meta_update, meta_steps=args.meta_steps)

# Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# Define model evaluation function
def evaluate(features, support, placeholders, max_graph_size, max_num_edges):
    t_test = time.time()
    feed_dict_val = construct_feed_dict4pred(features, support, placeholders)
    feed_dict_val.update({placeholders['max_graph_size']: max_graph_size})
    feed_dict_val.update({placeholders['max_num_edges']: max_num_edges})
    outs_val = sess.run([model.outputs_softmax], feed_dict=feed_dict_val)
    return (time.time() - t_test), outs_val[0]

def findNodeEdges(adj):
    nn = adj.shape[0]
    edges = []
    for i in range(nn):
        edges.append(adj.indices[adj.indptr[i]:adj.indptr[i+1]])
    return edges

def isis_v2(edges, nIS_vec_local, cn):
    return np.sum(nIS_vec_local[edges[cn]] == 1) > 0

def isis(edges, nIS_vec_local):
    tmp = (nIS_vec_local==1)
    return np.sum(tmp[edges[0]]*tmp[edges[1]]) > 0

# def add_rnd_q(cns, nIS_vec_local):
#     global adj_0
#
#     nIS_vec_local[cns] = 1
#     tmp = sp.find(adj_0[cns, :] == 1)
#     nIS_vec_local[tmp[1]] = 0
#     remain_vec_tmp = (nIS_vec_local == -1)
#     adj = adj_0
#     adj = adj[remain_vec_tmp, :]
#     adj = adj[:, remain_vec_tmp]
#     if reduce_graph(adj, nIS_vec_local):
#         return True
    # return False


def fake_reduce_graph(adj):
    reduced_node = -np.ones(adj.shape[0])
    reduced_adj = adj
    mapping = np.arange(adj.shape[0])
    reverse_mapping = np.arange(adj.shape[0])
    crt_is_size = 0
    return reduced_node, reduced_adj, mapping, reverse_mapping, crt_is_size


def fake_local_search(adj, nIS_vec):
    return nIS_vec.astype(int)


def reduce_graph(adj, nIS_vec_local, diver_id=-1):
    global best_IS_num
    global best_IS_vec
    global bsf_q
    global adj_0
    global q_ct
    global id
    global out_id
    global res_ct

    remain_vec = (nIS_vec_local == -1)
    # reduce graph
    if args.reduction:
        reduced_node, reduced_adj, mapping, reverse_mapping, crt_is_size = api.reduce_graph(adj)
    else:
        reduced_node, reduced_adj, mapping, reverse_mapping, crt_is_size = fake_reduce_graph(adj)
    nIS_vec_sub = reduced_node.copy()
    nIS_vec_sub_tmp = reduced_node.copy()
    nIS_vec_sub[nIS_vec_sub_tmp == 0] = 1
    nIS_vec_sub[nIS_vec_sub_tmp == 1] = 0
    reduced_nn = reduced_adj.shape[0]

    # update MIS after reduction
    tmp = sp.find(adj[nIS_vec_sub == 1, :] == 1)
    nIS_vec_sub[tmp[1]] = 0
    nIS_vec_local[remain_vec] = nIS_vec_sub
    nIS_vec_local[nIS_vec_local == 2] = -1

    # if the whole graph is reduced, we find a candidate
    if reduced_nn == 0:
        remain_vec_tmp = (nIS_vec_local == -1)
        if np.sum(remain_vec_tmp) == 0:
            print("find a candidate, full reduction.")
            # get a solution
            res_ct += 1
            if args.local_search:
                nIS_vec_local = api.local_search(adj_0, nIS_vec_local)
            else:
                nIS_vec_local = fake_local_search(adj_0, nIS_vec_local)
            if np.sum(nIS_vec_local) > best_IS_num:
                best_IS_num = np.sum(nIS_vec_local)
                best_IS_vec = deepcopy(nIS_vec_local)
                sio.savemat(args.output + '/res_%04d/%s' % (
                    time_limit, val_mat_names[id]), {'er_graph': adj_0, 'nIS_vec': best_IS_vec})
                statistics.collector.current_collector.collect_result(np.flatnonzero(best_IS_vec))
                print("find best IS vec.")
            print("ID: %03d" % id, "QItem: %03d" % q_ct, "Res#: %03d" % res_ct,
                  "Current: %d" % (np.sum(nIS_vec_local)), "Best: %d" % best_IS_num, "Reduction")
            return True
        else:
            adj = adj_0
            adj = adj[remain_vec_tmp, :]
            adj = adj[:, remain_vec_tmp]
            bsf_q.append([adj, nIS_vec_local.copy(), remain_vec.copy(), reduced_adj, reverse_mapping.copy(), mapping.copy()])
    else:
        bsf_q.append([adj, nIS_vec_local.copy(), remain_vec.copy(), reduced_adj, reverse_mapping.copy(), mapping.copy()])

    return False

# Init variables


if args.dimes:
    saver = tf.train.Checkpoint(optimizer=model.optimizer, variables=tf.trainable_variables())
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.latest_checkpoint(args.pretrained_weights)
    print(f"Restoring pretrained weights from {args.pretrained_weights}: {ckpt}")
    status = saver.restore(ckpt)
    status.run_restore_ops(session=sess)
else:
    saver = tf.train.Saver(max_to_keep=1000)
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(args.pretrained_weights)
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)


noout = FLAGS.diver_num # number of outputs
time_limit = args.time_limit # time limit for searching

if not os.path.isdir(args.output + "/res_%04d"%time_limit):
    os.makedirs(args.output + "/res_%04d"%time_limit)

# for graph reduction and local search
if args.reduction or args.local_search:
    api = reducelib()

import scipy

gurobi_mis = []

max_graph_size = 0
max_num_edges = 0

for id in range(len(val_mat_names)):
    stat_collector = statistics.collector.new_collector(val_mat_names[id])
    stat_collector.start_timer()
    stat_collector.start_process_timer()
    best_IS_num = -1
    mat_contents = sio.loadmat(data_path + '/' + val_mat_names[id])
    adj_0 = mat_contents['adj']

    output_file = args.output + '/res_%04d/%s' % (time_limit, val_mat_names[id])
    if os.path.isfile(output_file):
        print("%s exists, skip" % output_file)
        continue

    max_graph_size = max(max_graph_size, adj_0.shape[0] + 100)
    max_num_edges = max(max_num_edges, adj_0.count_nonzero() + adj_0.shape[0] + 100)

    # if args.self_loops:
    #     identity = scipy.sparse.identity(adj_0.shape[0], dtype=adj_0.dtype, format=adj_0.format)
    #     adj_0 = adj_0 + identity

    labels_given = False
    if 'indset_label' in mat_contents.keys():
        yy = mat_contents['indset_label']
        opt_num = np.sum(yy[:, 0])
        gurobi_mis.append(opt_num)
        labels_given = True
        print("Labels were given, terminating if optimal MIS found", file=sys.stderr)

    # edges_0 = sp.find(adj_0) # for isis version 1
    edges_0 = findNodeEdges(adj_0)
    nn = adj_0.shape[0]
    bsf_q = []
    q_ct = 0
    res_ct = 0
    out_id = -1

    start_time = time.time()
    if labels_given:
        nIS_vec_local = mat_contents['indset_label']
        print("opt_num", opt_num, labels_given, best_IS_num)

    # if labels_given and best_IS_num == opt_num:
    #     break

    if len(bsf_q) == 0:
        if q_ct == 0:
            if reduce_graph(adj_0, -np.ones(nn)):
                print("early break")
                continue
        else:
            continue

    # q_item = bsf_q.pop(len(bsf_q)-1)
    # q_item = bsf_q.pop(np.random.randint(max(len(bsf_q)-32, 0),len(bsf_q)))
    q_item = bsf_q.pop(np.random.randint(len(bsf_q)))
    q_ct += 1

    adj = q_item[0]
    remain_vec = deepcopy(q_item[2])
    reduced_adj = q_item[3]
    reverse_mapping = deepcopy(q_item[4])
    mapping = deepcopy(q_item[5])
    remain_nn = adj.shape[0]
    reduced_nn = reduced_adj.shape[0]
    assert np.allclose(adj.todense(), adj.todense().T, rtol=1e-05, atol=1e-08)

    print("reduced_nn", reduced_nn, len(bsf_q))
    if reduced_nn != 0:
        # GCN
        features = np.ones([reduced_nn, N_bd])
        features = sp.lil_matrix(features)
        features = preprocess_features(features)

        tmp_reduced_adj = reduced_adj
        if args.self_loops:
            identity = scipy.sparse.identity(reduced_adj.shape[0],
                                             dtype=reduced_adj.dtype, format=reduced_adj.format)
            reduced_adj = reduced_adj + identity
        support = simple_polynomials(reduced_adj, FLAGS.max_degree)
        _, z_out = evaluate(features, support, placeholders, max_graph_size, max_num_edges)
        z_out = z_out[:, 1::2] - z_out[:, 0::2]
        # z_out = z_out - np.mean(z_out, axis=0, keepdims=True)
        # z_out = z_out / np.std(z_out, axis=0, keepdims=True)
        # z_out = z_out * 2.0
        # print("z_out", z_out)
        stat_collector.add_iteration()

        # local search
        a, b, c = scipy.sparse.find(reduced_adj)
        graph_indices = np.stack((a, b), axis=1)
        count = 0
        start_time = time.time()
        while time.time() - start_time < time_limit:
            if count == 0:
                solution = reinforce.np_get_mis_solution(graph_indices, z_out, FLAGS.diver_num, max_graph_size,
                                                         max_num_edges)
            else:
                solution = reinforce.np_get_mis_solution(graph_indices, z_out, FLAGS.diver_num, max_graph_size,
                                                         max_num_edges, use_sample=True)
                if time_limit == 1:
                    break
            count += 1
            for out_id in range(solution.shape[0]):
                # if labels_given and best_IS_num == opt_num:
                #     break

                nIS_vec = deepcopy(q_item[1])
                # nIS_Prob_sub_t = z_out[:, 2 * out_id + 1]
                # nIS_Prob_sub = np.zeros(remain_nn)
                # nIS_Prob_sub[reverse_mapping] = nIS_Prob_sub_t
                # nIS_Prob = np.zeros(nn)
                # nIS_Prob[remain_vec] = nIS_Prob_sub
                #
                # # chosen nodes
                # cns_sorted = np.argsort(1 - nIS_Prob)

                # tt = time.time()
                nIS_vec_tmp = deepcopy(nIS_vec)
                cns = reverse_mapping[np.nonzero(solution[out_id])[0]]
                nIS_vec_tmp[cns] = 1

                # for cid in range(nn):
                #     if time.time()-start_time > time_limit:
                #         break
                #
                #     cn = cns_sorted[cid]
                #     # check graph
                #     if isis_v2(edges_0, nIS_vec_tmp, cn):
                #         pass
                #     else:
                #         nIS_vec_tmp[cn] = 1
                #         cns.append(cn)
                #         add_rnd_q(cns_sorted[:(cid+1)], deepcopy(nIS_vec))

                # print("time=", "{:.5f}".format((time.time() - tt)))

                nIS_vec[cns] = 1
                tmp = sp.find(adj_0[cns, :] == 1)
                nIS_vec[tmp[1]] = 0

                # nIS_vec = np.zeros_like(nIS_vec)
                # assert np.sum(nIS_vec == -1) == cns.shape[0]
                remain_vec_tmp = (nIS_vec == -1)

                print(np.sum(nIS_vec == 1), np.sum(remain_vec_tmp))
                if np.sum(remain_vec_tmp) == 0:
                    # get a solution
                    res_ct += 1
                    if args.local_search:
                        nIS_vec = api.local_search(adj_0, nIS_vec)
                    else:
                        nIS_vec = fake_local_search(adj_0, nIS_vec)
                    if np.sum(nIS_vec) > best_IS_num:
                        best_IS_num = np.sum(nIS_vec)
                        best_IS_vec = deepcopy(nIS_vec)
                        sio.savemat(args.output + '/res_%04d/%s' % (
                        time_limit, val_mat_names[id]), {'er_graph': adj_0, 'nIS_vec': best_IS_vec})
                        statistics.collector.current_collector.collect_result(np.flatnonzero(best_IS_vec))
                    print("ID: %03d" % id, "QItem: %03d" % q_ct, "Res#: %03d" % res_ct,
                          "Current: %d" % (np.sum(nIS_vec)),
                          "Best: %d" % best_IS_num, "Network", "Optimal: %d" % opt_num if labels_given else "")
                    continue
                adj = adj_0
                adj = adj[remain_vec_tmp, :]
                adj = adj[:, remain_vec_tmp]

                reduce_graph(adj, nIS_vec)
        else:
            nIS_vec = deepcopy(q_item[1])
            # if reduce_graph(adj, nIS_vec):
            #     continue
    try:
        print("best_IS_vec", np.sum(best_IS_vec))
        sio.savemat(args.output + '/res_%04d/%s' % (time_limit, val_mat_names[id]), {'er_graph': adj_0, 'nIS_vec': best_IS_vec})
    except Exception as e:
        print("Error while saving matrix")
        print(e)

statistics.collector.finalize(args.output + "/results.json")
if len(gurobi_mis):
    print(sum(gurobi_mis) / len(gurobi_mis))
