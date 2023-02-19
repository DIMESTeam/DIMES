# LibTorch Sampler for TSP

## Installation

Before installing our TSP sampler, please ensure **PyTorch Scatter 2.0.7** is already installed.

**Notice:** According to [this issue](https://github.com/DIMESTeam/DIMES/issues/2#issuecomment-1375406648), some later versions of PyTorch Scatter have different function names so they are incompatible with our TSP sampler.

Then, install our TSP sampler:

```bash
pip install ./torch_sampling
```

## Usage

Please refer to the code of TSP-KNN.

**Notice:** If you see the following warning when running our code, just ignore it.

```
[W OperatorEntry.cpp:111] Warning: Registering a kernel (registered by RegisterOperators) for operator torch_scatter::segment_sum_csr for dispatch key (catch all) that overwrote a previously registered kernel with the same dispatch key for the same operator. (function registerKernel)
```
