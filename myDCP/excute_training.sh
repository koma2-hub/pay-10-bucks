#!/bin/bash
echo "Hello, Ubuntu!"
echo "Train with intensity model!"
python main.py --exp_name with_intensity_trans5 --emb_nn dgcnnv2 --data_path /mnt/c/Users/komatsu/SICK/pay-10-bucks/myDCP/dataset_trans5 --use_intensity
echo "Train without intensity model!"
python main.py --exp_name without_intensity_trans5 --emb_nn dgcnnv2 --data_path /mnt/c/Users/komatsu/SICK/pay-10-bucks/myDCP/dataset_trans5