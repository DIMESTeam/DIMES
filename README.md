# DIMES: A Differentiable Meta Solver for Combinatorial Optimization Problems

[![NeurIPS](https://img.shields.io/badge/2022-NeurIPS-purple)](https://openreview.net/forum?id=9u05zr0nhx)

A meta-learner for large-scale combinatorial optimization problems including the traveling salesman problem (TSP) and the maximum independent set problem (MIS).

If you use our code, please cite [our paper](https://openreview.net/forum?id=9u05zr0nhx):

```bibtex
@inproceedings{qiu2022dimes,
  title={{DIMES}: A differentiable meta solver for combinatorial optimization problems},
  author={Ruizhong Qiu and Zhiqing Sun and Yiming Yang},
  booktitle={Advances in Neural Information Processing Systems 35},
  year={2022}
}
```

## DIMES-TSP

The code for DIMES-TSP is in the directories `TSP/TSP-KNN` and `TSP/TSP-Full`. `TSP-KNN` employs KNN edge pruning, while `TSP-Full` runs on the full graph.

We use $N$ to denote the number of nodes and $K$ to denote the number of neighbors in KNN pruning.

### Dependencies

Our code was tested under the following dependencies:

- GCC 7.5.0 on Ubuntu 18.04
- CUDA 11.0
- PyTorch 1.7.0
- [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter) 2.0.7
- [PyTorch Sparse](https://github.com/rusty1s/pytorch_sparse) 0.6.9
- [PyTorch Cluster](https://github.com/rusty1s/pytorch_cluster) 1.5.9
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) 2.0.4

**Notice:** According to [issue #2](https://github.com/DIMESTeam/DIMES/issues/2) and [issue #4](https://github.com/DIMESTeam/DIMES/issues/4), some other versions of PyTorch Scatter are incompatible with our TSP sampler.

### Usage

**Notice:** If you see the following warning when running our code, just ignore it.

```
[W OperatorEntry.cpp:111] Warning: Registering a kernel (registered by RegisterOperators) for operator torch_scatter::segment_sum_csr for dispatch key (catch all) that overwrote a previously registered kernel with the same dispatch key for the same operator. (function registerKernel)
```

#### Installation

Before installing our TSP sampler, please ensure **PyTorch Scatter 2.0.7** is already installed.

Install our TSP sampler:

```bash
cd TSP
pip install ./torch_sampling
```

#### Reproduction

To reproduce our results, please run:

```bash
cd TSP/TSP-KNN
./tsp{N}_test_{decoder}.sh tsp{N}
```
where `decoder` can be `G` / `AS_G` / `S` / `AS_S` / `MCTS` / `AS_MCTS`.

#### Training

To train a model from scratch, please run:

```bash
cd TSP/TSP-KNN
./tsp{N}_train.sh
```

Output (parameters of trained models) will be saved in folder `output/`.

The output will have a prefix generated automatically. We call this prefix `save_name`. For example, `save_name` of the file `output/tsp500~net120.pt` is `tsp500`. A generated `save_name` is like `dimes-tsp{N}-knn{K}@{timestamp}`, where a timestamp is to ensure uniqueness of filenames.

#### Testing

To test a trained model, please run:

```bash
cd TSP/TSP-KNN
./tsp{N}_test_{decoder}.sh {save_name}
```

where `save_name` is the one generated during training, and `decoder` can be `G` / `AS_G` / `S` / `AS_S` / `MCTS` / `AS_MCTS`.

### Resources

#### Test Instances

The test instances are originally provided by [Fu et al. (2021)](https://github.com/Spider-SCNU/TSP). We have reformatted them for our code. Reformatted test instances are provided in folder `input/`.

#### Trained Models

Our trained models `tsp{N}_net*.pt` are in folder `output/`.

#### Heatmaps for MCTS

The predicted heatmaps of DIMES+MCTS and DIMES+AS+MCTS are in folder `output/`.

### URLs of Baselines

- EAN (Deudon et al., 2018): https://github.com/MichelDeudon/encode-attend-navigate
- AM (Kool et al., 2019): https://github.com/wouterkool/attention-learn-to-route
- GCN (Joshi et al., 2019): https://github.com/chaitjo/graph-convnet-tsp
- POMO (Kwon et al., 2020): https://github.com/yd-kwon/POMO
- EAS (Hottung et al., 2022): https://github.com/ahottung/EAS
- Att-GCN (Fu et al., 2021): https://github.com/Spider-SCNU/TSP (We use their CPU-version MCTS code.)

## DIMES-MIS

The code for DIMES-MIS is in the directory `MIS`. Our code is adapted from the code of [mis-benchmark-framework
](https://github.com/MaxiBoether/mis-benchmark-framework)

### Usage

#### Installation

Before running our code, please install our MIS sampler:

```bash
cd MIS
conda env create -f environment.yml
```

#### Reproduction

```bash
cd MIS
bash scripts/solve_intel_dimes_er.sh
bash scripts/solve_intel_dimes_sat.sh
```

### Data

The evaluation data can be found at `MIS/data`.

#### SAT

For SATLIB (https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/descr_CBS.html), since the converted graphs are quite huge and the graphs are fixed. We only provide the file names for the test split, while the rest can be used for training.

#### ER Graphs

For ER graphs, since the graphs are randomly generated, we provide the gpickle version of the test graphs that we used for evaluation (in a shared Google drive link). The training graphs can be generated with another call of

```bash 
python -u main.py gendata \
    random \
    None \
    data_er/train \
    --model er \
    --min_n 700 \
    --max_n 800 \
    --num_graphs 16384 \
    --er_p 0.15
```

or

```bash 
python -u main.py gendata \
    random \
    None \
    data_er_large/train \
    --model er \
    --min_n 9000 \
    --max_n 11000 \
    --num_graphs 4096 \
    --er_p 0.02
```
