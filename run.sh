#! /bin/bash
python experiments/run_expt.py --dataset ogb-molhiv  --algorithm ERM --model gin_virtual_mol  --device 0 --root_dir data &