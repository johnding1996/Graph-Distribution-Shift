#! /bin/bash
python experiments/run_expt.py --dataset ogb-molhiv  --algorithm groupDRO --model gcn  --root_dir data 

python experiments/run_expt.py --dataset RotatedMNIST  --algorithm  groupDRO  --model gcn  --root_dir data  --frac 0.1