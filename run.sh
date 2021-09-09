#! /bin/bash
python experiments/run_expt.py --dataset ogb-molhiv  --algorithm ERM  --device 0 --root_dir data &
python experiments/run_expt.py --dataset ogb-molhiv  --algorithm groupDRO  --device 1   --root_dir data &
python experiments/run_expt.py --dataset ogb-molhiv  --algorithm deepCORAL  --device 2  --root_dir data &
python experiments/run_expt.py --dataset ogb-molhiv  --algorithm IRM  --device 3   --root_dir data &