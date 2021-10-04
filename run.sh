#! /bin/bash

python experiments/run_expt.py  --dataset ogb-molpcba    --algorithm GSN  --model gin --root_dir data --seed 0  --gsn_id_type cycle_graph   &

python experiments/run_expt.py  --dataset ogb-molpcba    --algorithm GSN  --model gin --root_dir data --seed 0  --gsn_id_type path_graph  &

python experiments/run_expt.py  --dataset ogb-molpcba    --algorithm GSN  --model gin --root_dir data --seed 0  --gsn_id_type complete_graph   &



python experiments/run_expt.py  --dataset RotatedMNIST   --algorithm GSN  --model gin --root_dir data --seed 0  --gsn_id_type path_graph  & 



# python experiments/run_expt.py  --dataset ogb-molpcba    --algorithm 3wlgnn  --model gin --root_dir data --seed 0  --gsn_id_type cycle_graph   --batch_size 64


# python experiments/run_expt.py  --dataset ogb-molpcba    --algorithm ERM   --model 3wlgnn --root_dir data --seed 0
