# DIMES: A Differentiable Meta Solver for Combinatorial Optimization Problems

A meta-learner for large-scale combinatorial optimization problems including the traveling salesman problem (TSP) and the maximum independent set problem (MIS)

## DIMES-TSP

The code for DIMES-TSP is in the directories `TSP/TSP-KNN` and `TSP/TSP-Full`. `TSP-KNN` employs KNN edge pruning, while TSP-Full runs on the full graph.

We use $N$ to denote the number of nodes.

### Usage

#### Installation

Before running our code, please install our TSP sampler:

```bash
cd TSP
pip install ./torch_sampling
```

#### Training

To train a model from scratch, please run:

```bash
./tsp{N}_train.sh
```

Parameters of trained models will be saved in folder `output/`.

The output files will have a common prefix generated automatically. We call this prefix `save_name`.

#### Testing

To test a trained model, please run:

```bash
./tsp{N}_test_{decoding}.sh {save_name}
```

where `save_name` is the one generated during training, and `decoding` can be `G`, `AS_G`, `S`, `AS_S`, `MCTS`, or `AS_MCTS`.

To reproduce our results, please let `save_name` be `tsp{N}` so that our trained model will be loaded.

### Resources

#### Test Instances

The test instances are originally provided by [Fu et al. (2021)](https://github.com/Spider-SCNU/TSP). We have reformatted them for our code. Reformatted test instances are provided in folder `input`.

#### Trained Models

Our trained models `tsp{N}_net*.pt` are in folder `output/`. Please put the reformatted test instances in folder `models/`.

### Dependencies

- CUDA 11.0
- PyTorch 1.7.0
- PyTorch Scatter 2.0.7
- PyTorch Sparse 0.6.9
- PyTorch Cluster 1.5.9
- PyTorch Geometric 2.0.4

### URLs for Baselines

- EAN (Deudon et al., 2018): https://github.com/MichelDeudon/encode-attend-navigate
- AM (Kool et al., 2019): https://github.com/wouterkool/attention-learn-to-route
- GCN (Joshi et al., 2019): https://github.com/chaitjo/graph-convnet-tsp
- Att-GCN (Fu et al., 2021): https://github.com/Spider-SCNU/TSP

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
