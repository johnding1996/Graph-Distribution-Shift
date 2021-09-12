#! /bin/bash


for d in 'ogb-molhiv' 'RotatedMNIST' 
    do
        for m in  'gcn' 'gcn_virtual' 'gin' 'gin_virtual' 
            do 
                python experiments/run_expt.py --dataset ${d} --algorithm ERM --model ${m}  --root_dir data  --device 0 --eval_only True &
                python experiments/run_expt.py --dataset ${d}  --algorithm deepCORAL --model ${m}  --root_dir data   --device 1  --eval_only True &
                python experiments/run_expt.py --dataset ${d}   --algorithm groupDRO --model ${m}  --root_dir data   --device 2  --eval_only True&
                python experiments/run_expt.py --dataset ${d}  --algorithm IRM --model ${m}  --root_dir data   --device 3  --eval_only True &
                wait
            done
    done

python experiments/run_expt.py --dataset ogb-molhiv --algorithm ERM --model gcn_virtual  --root_dir data  --device 0 
