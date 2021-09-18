#! /bin/bash
# python experiments/run_expt.py  --dataset ogb-molhiv --algorithm ERM --model gin --root_dir data  --gsn True 
# python experiments/run_expt.py  --dataset ogbg-ppa --algorithm ERM --model gin --root_dir data  --gsn True 

python experiments/run_expt.py  --dataset ogb-molhiv   --algorithm ERM --model gin --root_dir data  --gsn True --id_type path_graph  & 

python experiments/run_expt.py  --dataset ogb-molhiv    --algorithm ERM --model gin --root_dir data  --gsn True --id_type complete_graph   & 


python experiments/run_expt.py  --dataset ogb-molpcba --algorithm ERM --model gin --root_dir data  --gsn True --id_type path_graph   &


python experiments/run_expt.py  --dataset ogb-molpcba --algorithm ERM --model gin --root_dir data  --gsn True --id_type complete_graph  &



# python experiments/run_expt.py  --dataset ogb-molhiv --algorithm IRM --model gin --root_dir data  



python experiments/run_expt.py  --dataset ogbg-ppa   --algorithm ERM --model gin --root_dir data  --gsn True --id_type path_graph 

