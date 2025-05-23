python3 ./test_AS_MCTS.py \
    --seed 123456789 \
    --device cuda:0 \
    --n_nodes 10000 \
    --knn_k 50 \
    --outer_opt AdamW \
    --outer_opt_lr 0.001 \
    --outer_opt_wd 1e-5 \
    --inner_opt AdamW \
    --inner_opt_lr 0.01 \
    --inner_opt_wd 0.0001 \
    --net_units 32 \
    --net_act silu \
    --emb_agg mean \
    --emb_depth 12 \
    --par_depth 3 \
    --te_net 50 \
    --te_range_l 0 \
    --te_range_r 16 \
    --te_batch_size 10 \
    --te_tune_steps 50 \
    --te_tune_sample_size 1000 \
    --te_sample_size 1000 \
    --te_sample_tau 1.0 \
    --save_name $1

cd MCTS_10000 && ./solve.sh 10000 16 16
