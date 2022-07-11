mkdir -p /tmp/gpus
touch /tmp/gpus/.lock
touch /tmp/gpus/0.gpu

export XLA_PYTHON_CLIENT_PREALLOCATE=false
conda activate mwis-benchmark

python -u main.py \
    --self_loops --dimes train \
    intel-treesearch \
    data_er/train \
    data_er/output_dimes \
    --pretrained_weights data_er/output_dimes \
    --num_cuda_devices 1 \
    --normalize_factor 1.0 \
    --max_nodes 1400

