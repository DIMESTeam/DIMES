import functools

import jax
import jax.numpy as jnp
import jax.scipy
import numpy as np
import sys
import time


def sample(carry):
    (jax_flag, jax_solution, graph_indices, par, par_grad, rng_key) = carry
    prob = jax.nn.softmax(par, axis=-1)
    p_next_node = jax.random.categorical(rng_key, par)
    flag = (par[p_next_node] > -1e8)
    jax_solution = jax_solution.at[p_next_node].set(jnp.where(flag, 1, jax_solution[p_next_node]))

    # nn_idx1 = graph_indices[:, 1] == p_next_node
    # nn_idx1 = jnp.where(nn_idx1, graph_indices[:, 0] + 1, 0)
    nn_idx2 = graph_indices[:, 0] == p_next_node
    nn_idx2 = jnp.where(nn_idx2, graph_indices[:, 1] + 1, 0)

    new_par = jnp.concatenate([jnp.zeros_like(par[:1]), par], axis=0)
    new_par = new_par.at[p_next_node + 1].set(-1e9)
    # new_par = new_par.at[nn_idx1].set(-1e9)
    new_par = new_par.at[nn_idx2].set(-1e9)
    par = new_par[1:]

    par_grad = par_grad.at[p_next_node].add(jnp.where(flag, 1.0, 0.0))
    par_grad = par_grad - jnp.where(flag, prob, 0.0)
    rng_key, new_rng_key = jax.random.split(rng_key)
    jax_flag = jnp.logical_and(jax_flag, flag)

    new_carry = (jax_flag, jax_solution, graph_indices, par, par_grad, new_rng_key)
    return new_carry


def softmax_sample(jax_flag, jax_solution, graph_indices, par, par_grad, rng_key, graph_size):
    vmap_sample = jax.vmap(sample, in_axes=((0, 0, None, 0, 0, 0),), out_axes=(0, 0, None, 0, 0, 0))
    return jax.lax.while_loop(lambda carry: jnp.any(carry[0]),
                              vmap_sample,
                              (jax_flag, jax_solution, graph_indices, par, par_grad, rng_key))


def sample_with_log_prob(carry):
    (jax_flag, jax_solution, graph_indices, par, log_prob, jax_path, rng_key, step) = carry
    new_log_prob = jax.nn.log_softmax(par, axis=-1)
    p_next_node = jax.random.categorical(rng_key, par)
    flag = (par[p_next_node] > -1e8)
    jax_path = jax_path.at[step].set(p_next_node)
    log_prob = log_prob.at[step].set(jnp.where(flag, new_log_prob[p_next_node], 1.0))
    jax_solution = jax_solution.at[p_next_node].set(jnp.where(flag, 1, jax_solution[p_next_node]))

    # nn_idx1 = graph_indices[:, 1] == p_next_node
    # nn_idx1 = jnp.where(nn_idx1, graph_indices[:, 0] + 1, 0)
    nn_idx2 = graph_indices[:, 0] == p_next_node
    nn_idx2 = jnp.where(nn_idx2, graph_indices[:, 1] + 1, 0)

    new_par = jnp.concatenate([jnp.zeros_like(par[:1]), par], axis=0)
    new_par = new_par.at[p_next_node + 1].set(-1e9)
    # new_par = new_par.at[nn_idx1].set(-1e9)
    new_par = new_par.at[nn_idx2].set(-1e9)
    par = new_par[1:]
    rng_key, new_rng_key = jax.random.split(rng_key)
    jax_flag = jnp.logical_and(jax_flag, flag)

    new_carry = (jax_flag, jax_solution, graph_indices, par, log_prob, jax_path, new_rng_key, step + 1)
    return new_carry


def softmax_sample_log_prob(jax_flag, jax_solution, graph_indices, par, log_prob, jax_path, rng_key, graph_size):
    vmap_sample = jax.vmap(sample_with_log_prob, in_axes=((0, 0, None, 0, 0, 0, 0, None),),
                           out_axes=(0, 0, None, 0, 0, 0, 0, None))
    return jax.lax.while_loop(lambda carry: jnp.any(carry[0]),
                              vmap_sample,
                              (jax_flag, jax_solution, graph_indices, par, log_prob, jax_path, rng_key, 0))


def sample_with_path(carry):
    (jax_flag, graph_indices, par, old_log_prob, par_grad, jax_path, old_reward, step) = carry
    prob = jax.nn.softmax(par, axis=-1)
    new_log_prob = jax.nn.log_softmax(par, axis=-1)
    current_old_log_prob = old_log_prob[step]
    p_next_node = jax_path[step]
    new_log_prob = new_log_prob[p_next_node]
    flag = (par[p_next_node] > -1e8)

    # nn_idx1 = graph_indices[:, 1] == p_next_node
    # nn_idx1 = jnp.where(nn_idx1, graph_indices[:, 0] + 1, 0)
    nn_idx2 = graph_indices[:, 0] == p_next_node
    nn_idx2 = jnp.where(nn_idx2, graph_indices[:, 1] + 1, 0)

    new_par = jnp.concatenate([jnp.zeros_like(par[:1]), par], axis=0)
    new_par = new_par.at[p_next_node + 1].set(-1e9)
    # new_par = new_par.at[nn_idx1].set(-1e9)
    new_par = new_par.at[nn_idx2].set(-1e9)
    par = new_par[1:]

    reward = old_reward
    log_prob_diff = new_log_prob - current_old_log_prob
    reward = jnp.where(jnp.logical_and(log_prob_diff > 0.2, reward > 0), 0.0, reward)
    reward = jnp.where(jnp.logical_and(log_prob_diff < -0.2, reward < 0), 0.0, reward)
    reward = reward * jnp.exp(log_prob_diff)

    par_grad = par_grad.at[p_next_node].add(jnp.where(flag, 1.0, 0.0) * reward)
    par_grad = par_grad - jnp.where(flag, prob, 0.0) * reward
    jax_flag = jnp.logical_and(jax_flag, flag)

    new_carry = (jax_flag, graph_indices, par, old_log_prob, par_grad, jax_path, old_reward, step + 1)
    return new_carry


def softmax_sample_with_path(jax_flag, graph_indices, par, old_log_prob, par_grad, jax_path, old_reward, graph_size):
    vmap_sample = jax.vmap(sample_with_path, in_axes=((0, None, 0, 0, 0, 0, 0, None),),
                           out_axes=(0, None, 0, 0, 0, 0, 0, None))
    return jax.lax.while_loop(lambda carry: jnp.any(carry[0]),
                              vmap_sample,
                              (jax_flag, graph_indices, par, old_log_prob, par_grad, jax_path, old_reward, 0))


beam_size = 4096


@jax.jit
def get_merged_grad(graph_indices, par, rng_key):
    print("grad beam_size", beam_size)
    graph_size = par.shape[-1]
    par = jnp.tile(par.reshape((-1, 1, graph_size)), (1, beam_size, 1)).reshape((-1, graph_size))

    jax_solution = jnp.zeros(par.shape, dtype=jnp.int32)
    jax_flag = jnp.full(par.shape[0], True, dtype=jnp.bool_)
    par_grad = jnp.zeros_like(par)

    _, solutions, _, _, par_grad, _ = softmax_sample(
        jax_flag, jax_solution, graph_indices, par, par_grad, rng_key, graph_size)
    solutions = solutions.reshape((-1, beam_size, graph_size))
    par_grad = par_grad.reshape((-1, beam_size, graph_size))

    reward = jnp.asarray(jnp.sum(solutions, axis=-1, keepdims=True), jnp.float32)
    old_reward = reward
    reward = reward - jnp.mean(reward, axis=1, keepdims=True)
    reward = reward / (jnp.std(reward, axis=1, keepdims=True) + 1e-3)
    par_grad = par_grad * reward
    return par_grad.mean(axis=1), old_reward, jnp.abs(jnp.mean(par_grad, axis=1)), jnp.std(par_grad, axis=1)


batched_gradv2 = jax.pmap(get_merged_grad, in_axes=(None, 0, 0), out_axes=(0, 0, 0, 0))


@jax.jit
def get_merged_aux_grad(graph_indices, par, rng_key):
    print("aux_grad beam_size", beam_size)
    graph_size = par.shape[-1]
    old_par = par.reshape((-1, 1, graph_size))
    par = jnp.tile(old_par, (1, beam_size, 1)).reshape((-1, graph_size))

    jax_flag = jnp.full(par.shape[0], True, dtype=jnp.bool_)
    log_prob = jnp.zeros(par.shape, dtype=jnp.float32)
    jax_path = jnp.zeros(par.shape, dtype=jnp.int32)
    jax_solution = jnp.zeros(par.shape, dtype=jnp.int32)

    _, jax_solution, _, _, log_prob, jax_path, _, _ = softmax_sample_log_prob(
        jax_flag, jax_solution, graph_indices, par, log_prob, jax_path, rng_key, graph_size)
    log_prob = log_prob.reshape((-1, beam_size, graph_size))
    # jax_path = jax_path.reshape((-1, beam_size, graph_size))
    jax_solution = jax_solution.reshape((-1, beam_size, graph_size))
    jax_solution = jnp.asarray(jax_solution, dtype=jnp.float32)

    reward = jnp.sum(jax_solution, axis=-1, keepdims=True)
    old_reward = reward
    reward = reward - jnp.mean(reward, axis=1, keepdims=True)
    reward = reward / (jnp.std(reward, axis=1, keepdims=True) + 1e-3)

    log_prob_q = jnp.sum(log_prob, axis=-1, keepdims=True) + jax.scipy.special.gammaln(old_reward + 1)
    log_prob_p_score = jnp.sum(jax_solution * old_par, axis=-1, keepdims=True)
    log_prob_p_normalizer = jax.scipy.special.logsumexp(log_prob_p_score - log_prob_q, axis=1, keepdims=True)
    log_prob_p_normalizer = log_prob_p_normalizer - jnp.log(jax_solution.shape[1])
    log_prob_p = jnp.minimum(log_prob_p_score - log_prob_p_normalizer, 0.0)
    factor1 = jnp.exp(jnp.clip(log_prob_p - log_prob_q, -1.0, 1.0))
    # mean_log_prob_q = jnp.mean(log_prob_q, axis=1, keepdims=True)
    # mean_log_prob_p_score = jnp.mean(log_prob_p_score, axis=1, keepdims=True)
    # factor1 = jnp.exp(jnp.clip((log_prob_p_score - mean_log_prob_p_score) -
    #                            (log_prob_q - mean_log_prob_q), -1.0, 1.0))

    # log_expected_solution = jnp.where(jax_solution > 0.5, 0.0, -1e9) + log_prob_p_score - log_prob_q
    # log_expected_solution = jax.scipy.special.logsumexp(log_expected_solution, axis=1, keepdims=True)
    # log_expected_solution = log_expected_solution - jnp.log(jax_solution.shape[1]) - log_prob_p_normalizer
    # expected_solution = jnp.exp(jnp.maximum(log_expected_solution, 0.0))
    expected_solution = jnp.mean(jax_solution, axis=1, keepdims=True)
    par_grad = reward * factor1 * (jax_solution - expected_solution)
    return par_grad.mean(axis=1), old_reward, jnp.abs(jnp.mean(par_grad, axis=1)), jnp.std(par_grad, axis=1)


batched_aux_gradv2 = jax.pmap(get_merged_aux_grad, in_axes=(None, 0, 0), out_axes=(0, 0, 0, 0))


@jax.jit
def get_merged_sample(graph_indices, par, rng_key):
    print("sample beam_size", beam_size)
    graph_size = par.shape[-1]
    par = jnp.tile(par.reshape((-1, 1, graph_size)), (1, beam_size, 1)).reshape((-1, graph_size))

    jax_flag = jnp.full(par.shape[0], True, dtype=jnp.bool_)
    log_prob = jnp.zeros(par.shape, dtype=jnp.float32)
    jax_path = jnp.zeros(par.shape, dtype=jnp.int32)
    jax_solution = jnp.zeros(par.shape, dtype=jnp.int32)

    _, jax_solution, _, _, log_prob, jax_path, _, _ = softmax_sample_log_prob(
        jax_flag, jax_solution, graph_indices, par, log_prob, jax_path, rng_key, graph_size)
    log_prob = log_prob.reshape((-1, beam_size, graph_size))
    jax_path = jax_path.reshape((-1, beam_size, graph_size))
    jax_solution = jax_solution.reshape((-1, beam_size, graph_size))

    reward = jnp.asarray(jnp.sum(jax_solution, axis=-1), jnp.float32)
    reward = reward - jnp.mean(reward, axis=1, keepdims=True)
    reward = reward / (jnp.std(reward, axis=1, keepdims=True) + 1e-3)

    return log_prob, jax_path, reward


@jax.jit
def get_merged_best_sample(graph_indices, par, rng_key):
    print("sample beam_size", beam_size)
    graph_size = par.shape[-1]
    par = jnp.tile(par.reshape((-1, 1, graph_size)), (1, beam_size, 1)).reshape((-1, graph_size))

    jax_flag = jnp.full(par.shape[0], True, dtype=jnp.bool_)
    log_prob = jnp.zeros(par.shape, dtype=jnp.float32)
    jax_path = jnp.zeros(par.shape, dtype=jnp.int32)
    jax_solution = jnp.zeros(par.shape, dtype=jnp.int32)

    _, jax_solution, _, _, log_prob, _, _, _ = softmax_sample_log_prob(
        jax_flag, jax_solution, graph_indices, par, log_prob, jax_path, rng_key, graph_size)
    log_prob = log_prob.reshape((-1, beam_size, graph_size))
    jax_solution = jax_solution.reshape((-1, beam_size, graph_size))

    reward = jnp.asarray(jnp.sum(jax_solution, axis=-1), jnp.float32)
    best_idx = jnp.reshape(jnp.argmax(reward, axis=1), (-1, 1, 1))
    jax_path = jnp.take_along_axis(jax_solution, best_idx, axis=1)
    jax_path = jnp.squeeze(jax_path, axis=1)

    return log_prob, jax_path, reward


batched_samplev2 = jax.pmap(get_merged_sample, in_axes=(None, 0, 0), out_axes=(0, 0, 0))


batched_best_samplev2 = jax.pmap(get_merged_best_sample, in_axes=(None, 0, 0), out_axes=(0, 0, 0))


@jax.jit
def get_merged_grad_with_path(graph_indices, par, old_log_prob, jax_path, old_reward):
    print("grad_with_path beam_size", beam_size)
    graph_size = par.shape[-1]
    par = jnp.tile(par.reshape((-1, 1, graph_size)), (1, beam_size, 1)).reshape((-1, graph_size))
    old_log_prob = old_log_prob.reshape((-1, graph_size))
    jax_path = jax_path.reshape((-1, graph_size))
    old_reward = old_reward.reshape(-1)

    jax_flag = jnp.full(par.shape[0], True, dtype=jnp.bool_)
    par_grad = jnp.zeros_like(par)

    _, _, _, _, par_grad, _, _, _ = softmax_sample_with_path(
        jax_flag, graph_indices, par, old_log_prob, par_grad, jax_path, old_reward, graph_size)
    par_grad = par_grad.reshape((-1, beam_size, graph_size))
    par_grad = jnp.mean(par_grad, axis=1)
    return par_grad


batched_grad_with_pathv2 = jax.pmap(get_merged_grad_with_path, in_axes=(None, 0, 0, 0, 0), out_axes=0)


def np_get_merged_grad(graph_indices, par, diver_num, max_graph_size=1400, max_num_edges=15000, aux_grad=False):
    cur_time = time.time()
    graph_size = int(par.shape[0])

    seed = np.random.randint(sys.maxsize)
    rng_key = jax.random.PRNGKey(seed)
    rng_key = jax.random.split(rng_key, diver_num * beam_size)

    par = par.transpose().reshape((diver_num, graph_size))
    print(graph_indices.shape, par.shape)

    if graph_indices.shape[0] < max_num_edges:
        graph_indices = np.concatenate(
            [graph_indices, np.zeros((max_num_edges - graph_indices.shape[0], 2), dtype=np.int32)], axis=0)

    if par.shape[-1] < max_graph_size:
        par = np.concatenate([par, np.zeros((diver_num, max_graph_size - par.shape[-1])) - 1e9], axis=-1)

    # graph_indices = jax.device_put(graph_indices)
    # par = jax.device_put(par)

    def split_to_gpus(tensor, n_gpus=None):
        if n_gpus is None:
            n_gpus = jax.local_device_count()
        shape = tensor.shape
        return tensor.reshape((n_gpus, -1) + shape[1:])

    def merge_back_from_gpus(tensor):
        shape = tensor.shape
        return tensor.reshape((-1,) + shape[2:])

    second_time = time.time()
    if aux_grad:
        par_grad, old_reward, par_grad_abs_mean, par_grad_std = batched_aux_gradv2(
            graph_indices, split_to_gpus(par), split_to_gpus(rng_key))
    else:
        par_grad, old_reward, par_grad_abs_mean, par_grad_std = batched_gradv2(
            graph_indices, split_to_gpus(par), split_to_gpus(rng_key))
    third_time = time.time()
    output = (
        np.array(merge_back_from_gpus(par_grad).transpose()[:graph_size, :]),
        np.array(merge_back_from_gpus(old_reward)),
        np.mean(par_grad_abs_mean),
        np.mean(par_grad_std),
    )
    final_time = time.time()
    print(second_time - cur_time, third_time - second_time, final_time - third_time)
    return output


def np_get_merged_grad_ppo(graph_indices, par, old_log_prob, jax_path, old_reward,
                           diver_num, max_graph_size=1400, max_num_edges=15000):
    graph_size = int(par.shape[0])
    jax_path = jax_path.astype(np.int32)

    seed = np.random.randint(sys.maxsize)
    rng_key = jax.random.PRNGKey(seed)
    rng_key = jax.random.split(rng_key, diver_num * beam_size)

    par = par.transpose().reshape((diver_num, graph_size))
    print(graph_indices.shape, par.shape)

    if graph_indices.shape[0] < max_num_edges:
        graph_indices = np.concatenate([graph_indices, np.zeros((max_num_edges - graph_indices.shape[0], 2), dtype=np.int32)], axis=0)

    if par.shape[-1] < max_graph_size:
        par = np.concatenate([par, np.zeros((diver_num, max_graph_size - par.shape[-1])) - 1e9], axis=-1)

    if jax_path.shape[-1] < max_graph_size:
        jax_path = np.concatenate([jax_path, np.zeros(
            (diver_num, beam_size, max_graph_size - jax_path.shape[-1]), dtype=np.int32)], axis=-1)

    def split_to_gpus(tensor, n_gpus=None):
        if n_gpus is None:
            n_gpus = jax.local_device_count()
        shape = tensor.shape
        return tensor.reshape((n_gpus, -1) + shape[1:])

    def merge_back_from_gpus(tensor):
        shape = tensor.shape
        return tensor.reshape((-1,) + shape[2:])

    if np.sum(old_log_prob) > 0.0:
        old_log_prob, jax_path, old_reward = batched_samplev2(graph_indices, split_to_gpus(par), split_to_gpus(rng_key))
        old_log_prob = merge_back_from_gpus(old_log_prob)
        jax_path = merge_back_from_gpus(jax_path)
        old_reward = merge_back_from_gpus(old_reward)

    par_grad = batched_grad_with_pathv2(
        graph_indices, split_to_gpus(par),
        split_to_gpus(old_log_prob), split_to_gpus(jax_path), split_to_gpus(old_reward))
    par_grad = merge_back_from_gpus(par_grad).transpose()[:graph_size, :]
    jax_path = np.array(jax_path[:, :, :graph_size], dtype=np.int64)
    return par_grad, old_log_prob, jax_path, old_reward


def np_get_merged_grad_single(graph_indices, par, diver_num, max_graph_size=1400, max_num_edges=15000):
    graph_size = int(par.shape[0])

    seed = np.random.randint(32768)
    rng_key = jax.random.PRNGKey(seed)
    rng_key = jax.random.split(rng_key, diver_num * beam_size)

    par = par.transpose().reshape((diver_num, graph_size))

    print(graph_indices.shape, par.shape)

    if graph_indices.shape[0] < max_num_edges:
        graph_indices = np.concatenate([graph_indices, np.zeros((max_num_edges - graph_indices.shape[0], 2), dtype=np.int32)], axis=0)

    if par.shape[-1] < max_graph_size:
        par = np.concatenate([par, np.zeros((diver_num, max_graph_size - par.shape[-1])) - 1e9], axis=-1)

    graph_indices = jax.device_put(graph_indices)
    par = jax.device_put(par)

    par_grad = get_merged_grad(graph_indices, par, rng_key)
    return np.array(par_grad.transpose()[:graph_size, :])


def decode(carry):
    (jax_flag, jax_solution, graph_indices, par) = carry
    p_next_node = jnp.argmax(par, axis=-1)
    flag = (par[p_next_node] > -1e8)

    jax_solution = jax_solution.at[p_next_node].set(jnp.where(flag, 1, jax_solution[p_next_node]))
    # nn_idx1 = graph_indices[:, 1] == p_next_node
    # nn_idx1 = jnp.where(nn_idx1, graph_indices[:, 0] + 1, 0)
    imp_index = 1
    nn_idx2 = graph_indices[:, 0] == p_next_node
    nn_idx2 = jnp.where(nn_idx2, graph_indices[:, 1] + imp_index, 0)

    new_par = jnp.concatenate([jnp.zeros_like(par[:imp_index]), par], axis=0)
    new_par = new_par.at[p_next_node + imp_index].set(-1e9)
    # new_par = new_par.at[nn_idx1].set(-1e9)
    new_par = new_par.at[nn_idx2].set(-1e9)
    par = new_par[imp_index:]

    jax_flag = jnp.logical_and(jax_flag, flag)
    new_carry = (jax_flag, jax_solution, graph_indices, par)
    return new_carry


@jax.jit
def greedy_decode(graph_indices, par):
    jax_solution = jnp.zeros(par.shape, dtype=jnp.int32)
    jax_flag = jnp.full((par.shape[0], 1), True, dtype=jnp.bool_)
    vmap_decode = jax.vmap(decode, in_axes=((0, 0, None, 0),), out_axes=(0, 0, None, 0))
    return jax.lax.while_loop(lambda _: jnp.any(_[0]), vmap_decode,
                              (jax_flag, jax_solution, graph_indices, par))


def np_get_mis(graph_indices, par, diver_num, max_graph_size=1400, max_num_edges=15000):
    graph_size = int(par.shape[0])
    graph_indices = jax.device_put(graph_indices)
    par = jax.device_put(par).transpose().reshape((diver_num, graph_size))

    if graph_indices.shape[0] < max_num_edges:
        graph_indices = jnp.concatenate([graph_indices, jnp.zeros((max_num_edges - graph_indices.shape[0], 2), dtype=jnp.int32)], axis=0)

    if par.shape[-1] < max_graph_size:
        par = jnp.concatenate([par, jnp.zeros((diver_num, max_graph_size - par.shape[-1])) - 1e9], axis=-1)

    result = greedy_decode(graph_indices, par)[1]
    return np.array(jnp.sum(result, axis=-1), dtype=np.float32)


def np_get_mis_solution(graph_indices, par, diver_num, max_graph_size=1400, max_num_edges=15000, use_sample=False):
    graph_size = int(par.shape[0])
    graph_indices = jax.device_put(graph_indices)
    par = jax.device_put(par).transpose().reshape((diver_num, graph_size))

    if graph_indices.shape[0] < max_num_edges:
        graph_indices = jnp.concatenate([graph_indices, jnp.zeros((max_num_edges - graph_indices.shape[0], 2), dtype=jnp.int32)], axis=0)

    if par.shape[-1] < max_graph_size:
        par = jnp.concatenate([par, jnp.zeros((diver_num, max_graph_size - par.shape[-1])) - 1e9], axis=-1)

    if use_sample:
        def split_to_gpus(tensor, n_gpus=None):
            if n_gpus is None:
                n_gpus = jax.local_device_count()
            shape = tensor.shape
            return tensor.reshape((n_gpus, -1) + shape[1:])

        def merge_back_from_gpus(tensor):
            shape = tensor.shape
            return tensor.reshape((-1,) + shape[2:])

        seed = np.random.randint(sys.maxsize)
        rng_key = jax.random.PRNGKey(seed)
        rng_key = jax.random.split(rng_key, diver_num * beam_size)

        _, result, _ = batched_best_samplev2(graph_indices, split_to_gpus(par),
                                             split_to_gpus(rng_key))
        result = merge_back_from_gpus(result)
    else:
        _, result, _, _ = greedy_decode(graph_indices, par)
    return np.array(result, dtype=np.float32)
