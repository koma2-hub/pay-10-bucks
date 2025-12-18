#!/bin/bash
echo "Hello, Ubuntu!"
echo "Train with intensity model!"
python main.py --exp_name dcp_with_intensity_fullO --emb_nn dgcnnv2 --data_path /mnt/d/SICK/pay-10-bucks/myDCP/fulloverlap_dataset --use_intensity
echo "Train without intensity model!"
python main.py --exp_name dcp_without_intensity_fullO --emb_nn dgcnnv2 --data_path /mnt/d/SICK/pay-10-bucks/myDCP/fulloverlap_dataset