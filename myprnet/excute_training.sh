#!/bin/bash
echo "Hello, Ubuntu!"
echo "Train with intensity model!"
python main.py --exp_name prnet_with_intensity  --data_path /mnt/d/SICK/pay-10-bucks/myprnet/dataset --use_intensity --lr 0.0001
echo "Train without intensity model!"
python main.py --exp_name prnet_without_intensity  --data_path /mnt/d/SICK/pay-10-bucks/myprnet/dataset --lr 0.0001