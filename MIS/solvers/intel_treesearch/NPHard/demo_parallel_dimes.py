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
parser.add_argument("--reduction", action="store_true", default=False, help="Enable reduction of graph (kernelization).")
parser.add_argument("--random_heatmap", action="store_true", default=False, help="use random_heatmap.")
parser.add_argument("--local_search", action="store_true", default=False, help="Enable local search if time left.")
parser.add_argument("--model_prob_maps", type=int, action="store", default=32, help="Number of probability maps.")
parser.add_argument("--num_threads", type=int, action="store", default=16, help="Number of threads to use.")

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
import multiprocessing as mp
from multiprocessing import Manager, Value, Lock
from copy import deepcopy
import pickle
import tempfile
import shutil
from pathlib import Path

# import the libary for graph reduction and local search
if args.reduction or args.local_search:
    from reduce_lib import reducelib

from utils import *

import statistics
from functools import reduce
# test data path
data_path = args.input
val_mat_names = os.listdir(data_path)

# Define model evaluation function
def evaluate(sess, model, features, support, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict4pred(features, support, placeholders)
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

def add_rnd_q(cns, nIS_vec_local, pnum, lock, stat_collector):
    global adj_0

    nIS_vec_local[cns] = 1
    tmp = sp.find(adj_0[cns, :] == 1)
    nIS_vec_local[tmp[1]] = 0
    remain_vec_tmp = (nIS_vec_local == -1)
    adj = adj_0
    adj = adj[remain_vec_tmp, :]
    adj = adj[:, remain_vec_tmp]
    if reduce_graph(adj, nIS_vec_local, pnum, lock, stat_collector):
        return True
    return False

def fake_reduce_graph(adj):
    reduced_node = -np.ones(adj.shape[0])
    reduced_adj = adj
    mapping = np.arange(adj.shape[0])
    reverse_mapping = np.arange(adj.shape[0])
    crt_is_size = 0
    return reduced_node, reduced_adj, mapping, reverse_mapping, crt_is_size

def fake_local_search(adj, nIS_vec):
    return nIS_vec.astype(int)

def reduce_graph(adj, nIS_vec_local, pnum, lock, stat_collector):
    global best_IS_num
    global best_IS_vec
    global bsf_q
    global adj_0
    global q_ct
    global id
    global out_id
    global res_ct
    global sc

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
            # get a solution
            with lock:
                res_ct.value += 1
                local_res_ct = res_ct.value
            if args.local_search:
                nIS_vec_local = api.local_search(adj_0, nIS_vec_local)
            else:
                nIS_vec_local = fake_local_search(adj_0, nIS_vec_local)
            with lock:
                if np.sum(nIS_vec_local) > best_IS_num.value:
                    best_IS_num.value = np.sum(nIS_vec_local)
                    best_IS_vec = deepcopy(nIS_vec_local)
                    sio.savemat(args.output + '/res_%04d/%s' % (
                        time_limit, val_mat_names[id]), {'er_graph': adj_0, 'nIS_vec': best_IS_vec})
                    stat_collector.collect_result(np.flatnonzero(best_IS_vec))
            print("PID: %02d" % pnum, "ID: %03d" % id, "QItem: %03d" % q_ct.value, "Res#: %03d" % local_res_ct,
                    "Current: %d" % (np.sum(nIS_vec_local)), "Best: %d" % best_IS_num.value, "Reduction")
            return True
        adj = adj_0
        adj = adj[remain_vec_tmp, :]
        adj = adj[:, remain_vec_tmp]
        with lock:
            bsf_q.append([adj, nIS_vec_local.copy(), remain_vec.copy(), reduced_adj, reverse_mapping.copy()])
    else:
        with lock:
            bsf_q.append([adj, nIS_vec_local.copy(), remain_vec.copy(), reduced_adj, reverse_mapping.copy()])

    return False

def MPSearch(pnum, lock, stat_collector, pickle_path, labels_given):
    stat_collector.start_process_timer()

    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    from models import GCN_DEEP_DIVER_DIMES

    global best_IS_num  #
    global bsf_q #
    global q_ct #
    global res_ct #
    global best_IS_vec #

    global start_time
    global adj_0
    global opt_num
    global edges_0
    global nn
    global features_all
    global N_bd

    # Settings
    flags = tf.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 201, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
    flags.DEFINE_integer('diver_num', prob_maps, 'Number of outputs.')
    flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probaNUmbility).')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 1000, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', 1, 'Maximum Chebyshev polynomial degree.')
    flags.DEFINE_integer('num_layer', 20, 'number of layers.')

    # we need to define our argparse argument here aswell, otherwise tf.flags throws an exception
    flags.DEFINE_string("time_limit", "", "")
    flags.DEFINE_string("cuda_device", "", "")
    flags.DEFINE_boolean("self_loops", False, "")
    flags.DEFINE_boolean("reduction", False, "")
    flags.DEFINE_boolean("local_search", False, "")
    flags.DEFINE_boolean("random_heatmap", False, "")
    flags.DEFINE_string("model_prob_maps", "", "")
    flags.DEFINE_string("num_threads", "", "")

    # Some preprocessing

    num_supports = 1 + FLAGS.max_degree
    model_func = GCN_DEEP_DIVER_DIMES

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=(None, N_bd)),  # featureless: #points
        'labels': tf.placeholder(tf.float32, shape=(None, 2)),  # 0: not linked, 1:linked
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = model_func(placeholders, input_dim=N_bd, logging=True, random_heatmap=args.random_heatmap)

    # os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    # os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))
    # os.system('rm tmp')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)

    # Initialize session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # sess = tf.Session()

    # Init variables
    saver = tf.train.Saver(max_to_keep=1000)
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(args.pretrained_weights)
    print('%02d loaded' % pnum + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

    noout = FLAGS.diver_num  # number of outputs
    while time.time()-start_time < time_limit:
        if labels_given and best_IS_num.value == opt_num:
             break

        if len(bsf_q) == 0:
            if reduce_graph(adj_0, -np.ones(nn), pnum, lock, stat_collector):
                break

        with lock:
            if len(bsf_q) == 0:
                continue
            q_item = bsf_q.pop(np.random.randint(0,len(bsf_q)))
            q_ct.value += 1

        adj = q_item[0]
        remain_vec = deepcopy(q_item[2])
        reduced_adj = q_item[3]
        reverse_mapping = deepcopy(q_item[4])
        remain_nn = adj.shape[0]
        reduced_nn = reduced_adj.shape[0]

        if reduced_nn != 0:
            # GCN
            t1 = features_all[0][0:reduced_nn*N_bd,:]
            t2 = features_all[1][0:reduced_nn*N_bd]
            t3 = (reduced_nn, N_bd)
            features = (t1, t2, t3)
            support = simple_polynomials(reduced_adj, FLAGS.max_degree)

            _, z_out = evaluate(sess, model, features, support, placeholders)

            stat_collector.add_iteration()

            out_id = np.random.randint(noout)
            if labels_given and best_IS_num.value == opt_num:
                break

            nIS_vec = deepcopy(q_item[1])
            nIS_Prob_sub_t = z_out[:, 2 * out_id + 1]
            nIS_Prob_sub = np.zeros(remain_nn)
            nIS_Prob_sub[reverse_mapping] = nIS_Prob_sub_t
            nIS_Prob = np.zeros(nn)
            nIS_Prob[remain_vec] = nIS_Prob_sub

            # chosen nodes
            cns_sorted = np.argsort(1 - nIS_Prob)

            # tt = time.time()
            nIS_vec_tmp = deepcopy(nIS_vec)
            for cid in range(nn):
                cn = cns_sorted[cid]
                if isis_v2(edges_0, nIS_vec_tmp, cn):
                    break
                nIS_vec_tmp[cn] = 1
                # check graph
                if np.random.random_sample() > 0.7:
                    add_rnd_q(cns_sorted[:(cid + 1)], deepcopy(nIS_vec), pnum, lock, stat_collector)

            # print("time=", "{:.5f}".format((time.time() - tt)))

            cns = cns_sorted[:cid]
            nIS_vec[cns] = 1
            tmp = sp.find(adj_0[cns, :] == 1)
            nIS_vec[tmp[1]] = 0
            remain_vec_tmp = (nIS_vec == -1)

            if np.sum(remain_vec_tmp) == 0:
                # get a solution
                with lock:
                    res_ct.value += 1
                    local_res_ct = res_ct.value
                    if args.local_search:
                        nIS_vec = api.local_search(adj_0, nIS_vec)
                    else:
                        nIS_vec = fake_local_search(adj_0, nIS_vec)
                with lock:
                    if np.sum(nIS_vec) > best_IS_num.value:
                        best_IS_num.value = np.sum(nIS_vec)
                        best_IS_vec = deepcopy(nIS_vec)
                        sio.savemat(args.output + '/res_%04d/%s' % (
                        time_limit, val_mat_names[id]), {'er_graph': adj_0, 'nIS_vec': best_IS_vec})
                        stat_collector.collect_result(np.flatnonzero(best_IS_vec))
                print("PID: %02d" % pnum, "ID: %03d" % id, "QItem: %03d" % q_ct.value, "Res#: %03d" % local_res_ct,
                        "Current: %d" % (np.sum(nIS_vec)), "Best: %d" % best_IS_num.value, "Network")
                continue
            adj = adj_0
            adj = adj[remain_vec_tmp, :]
            adj = adj[:, remain_vec_tmp]

            if reduce_graph(adj, nIS_vec, pnum, lock, stat_collector):
                continue
        else:
            nIS_vec = deepcopy(q_item[1])
            if reduce_graph(adj, nIS_vec, pnum, lock, stat_collector):
                continue
    stat_collector.stop_timer()
    # pickle stat collector
    path = pickle_path / f"{pnum}.pickle"
    print(f"pickling into {path}")
    with open(path, 'wb') as f:
        pickle.dump(stat_collector, f)

time_limit = args.time_limit  # time limit for searching

if not os.path.isdir(args.output + "/res_%04d"%time_limit):
    os.makedirs(args.output + "/res_%04d"%time_limit)

# for graph reduction and local search
if args.local_search or args.reduction:
    api = reducelib()

if args.self_loops:
    import scipy

for id in range(len(val_mat_names)):

    manager = Manager()
    bsf_q = manager.list()
    q_ct = Value('i', 0)
    res_ct = Value('i', 0)
    best_IS_num = Value('i', -1)
    best_IS_vec = []

    lock = Lock()
    stat_collector = statistics.GraphResultCollector(val_mat_names[id])
    stat_collector.start_timer()
    mat_contents = sio.loadmat(data_path + '/' + val_mat_names[id])
    adj_0 = mat_contents['adj']
    if args.self_loops:
        identity = scipy.sparse.identity(adj_0.shape[0], dtype=adj_0.dtype, format=adj_0.format)
        adj_0 = adj_0 + identity

    labels_given = False
    if 'indset_label' in mat_contents.keys():
        yy = mat_contents['indset_label']
        opt_num = np.sum(yy[:,0])
        labels_given = True
        print("Labels were given, terminating if optimal MIS found", file=sys.stderr)

    # edges_0 = sp.find(adj_0) # for isis version 1
    edges_0 = findNodeEdges(adj_0)
    nn = adj_0.shape[0]
    N_bd = 32

    # process features and save them in advance
    features_all = np.ones([nn, N_bd])
    features_all = sp.lil_matrix(features_all)
    features_all = preprocess_features(features_all)

    start_time = time.time()

    pickle_path = Path(tempfile.mkdtemp())
    processes = [mp.Process(target=MPSearch, args=(pnum, lock, stat_collector, pickle_path, labels_given)) for pnum in range(args.num_threads)]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    rclist = []

    # todo fetch results here
    for f_path in pickle_path.rglob("*.pickle"):
        print(f"Reading file {str(f_path)} into result list.")
        with open(f_path, 'rb') as f:
            result = pickle.load(f)
        rclist.append(result)

    if len(rclist) != args.num_threads:
        print(f"Read {len(rclist)} results, but we had {args.num_threads} threads. Something is off!")

    shutil.rmtree(pickle_path)
    statistics.collector.collectors.append(reduce(lambda x, y: x + y, rclist))

    print(time.time() - start_time)

    # sio.savemat('result_IS4SAT_deep_ld32_c32_l20_cheb1_diver32_res32/res_tbf_mp_e_satlib_%04d/%s' % (time_limit, val_mat_names[id]),
    #                 {'er_graph': adj_0, 'nIS_vec': best_IS_vec})

statistics.collector.finalize(args.output + "/results.json")
