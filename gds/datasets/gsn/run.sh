# python gsn.py --seed 0 --onesplit True --dataset_name ogbg-molhiv  --features_scope full --vn True --id_type cycle_graph --induced True --k 6 --id_scope local --id_encoding one_hot_unique --id_embedding embedding --input_node_encoder atom_encoder --edge_encoder bond_encoder  --input_vn_encoder embedding --model_name GSN_edge_sparse_ogb --msg_kind ogb --num_layers 5 --d_out 300 --d_h 600 --dropout_features 0.5 --final_projection False --jk_mlp False --readout mean --batch_size 32 --num_epochs 100 --lr 1e-3 --scheduler None --loss_fn BCEWithLogitsLoss --prediction_fn multi_class_accuracy  --mode train  --device_idx 0 --wandb False



python gsn.py --seed 0  --dataset_name ogbg-molhiv --id_type cycle_graph --induced True --k 6 --id_scope local  
