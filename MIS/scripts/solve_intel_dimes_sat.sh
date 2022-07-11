mkdir -p /tmp/gpus
touch /tmp/gpus/.lock
touch /tmp/gpus/0.gpu

export XLA_PYTHON_CLIENT_PREALLOCATE=false
conda activate mwis-benchmark

python -u main.py \
    --self_loops --dimes train \
    intel-treesearch \
    data_sat/train_mis \
    data_sat/output_dimes \
    --pretrained_weights data_sat/output_dimes \
    --num_cuda_devices 1 \
    --normalize_factor 1.5 \
    --max_nodes 1400
