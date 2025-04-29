# LibTorch Sampler for TSP

## Installation

Before installing our TSP sampler, please ensure **PyTorch Scatter 2.0.7** is already installed.

**Note:** According to [issue #2](https://github.com/DIMESTeam/DIMES/issues/2) and [issue #4](https://github.com/DIMESTeam/DIMES/issues/4), some other versions of PyTorch Scatter can be incompatible with our TSP sampler.

For your reference, we used the following commands to install dependencies:

```bash
pip install --no-index torch-scatter==2.0.7 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-geometric==2.0.4
```

Install our TSP sampler:

```bash
cd TSP
pip install ./torch_sampling
```

**Note:** If the installer fails to find `-l_segment_csr_cpu` or `-l_segment_csr_cuda`, you may have to add their path to `torch_sampling/setup.py` manually. Please refer to [issue \#4](https://github.com/DIMESTeam/DIMES/issues/4#issuecomment-1863087703) for detail. We thank [@L-fu-des22](https://github.com/L-fu-des22) for sharing the experience about this issue.

## Usage

Please refer to the code of TSP-KNN.

**Note:** If you see the following warning when running our code, just ignore it.

```
[W OperatorEntry.cpp:111] Warning: Registering a kernel (registered by RegisterOperators) for operator torch_scatter::segment_sum_csr for dispatch key (catch all) that overwrote a previously registered kernel with the same dispatch key for the same operator. (function registerKernel)
```
