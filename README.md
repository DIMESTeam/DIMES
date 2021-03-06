# DIMES: A Differentiable Meta Solver for Combinatorial Optimization Problems

A meta-learner for large-scale combinatorial optimization problems including the traveling salesman problem (TSP) and the maximum independent set problem (MIS)

## DIMES-TSP

The code for DIMES-TSP is in the directory `TSP`. We use $N$ to denote the number of nodes.

### Usage

#### Installation

Before running our code, please install our TSP sampler:

```bash
cd TSP
pip install ./torch_sampling
```

#### Reproduction

To reproduce our results, please run:

```bash
./test_tsp{N}.sh
```

Our trained models will be used to predict heatmaps of test instances.

#### Training

To train a model from scratch, please run:

```bash
./train_tsp{N}.sh
```

Parameters of trained models will be saved in folder `models/`, and its file name will include a timestamp.

### Data

#### Trained Models

Parameters of our trained models are provided [here](@). Please put the reformatted test instances in folder `models/`.

#### Test Instances

The test instances are originally provided by [Fu et al. (2021)](https://github.com/Spider-SCNU/TSP). We have reformated them for our code. Reformatted test instances are provided [here](@). Please put the reformatted test instances in folder `input/`.

#### Predicted Heatmaps

Predicted heatmaps of test instances are provided [here](@).

### Dependencies

- CUDA 11.0
- PyTorch 1.7.0
- PyTorch Scatter 2.0.7
- PyTorch Sparse 0.6.9
- PyTorch Cluster 1.5.9
- PyTorch Geometric 2.0.4

### Baselines

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
