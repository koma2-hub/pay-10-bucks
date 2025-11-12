#!/bin/bash
echo "Hello, Ubuntu!"
python main.py --exp_name with_intensity --data_path /mnt/d/SICK/pay-10-bucks/myDCP/dataset/ --use_intensity True 
python main.py --exp_name without_intensity --data_path /mnt/d/SICK/pay-10-bucks/myDCP/dataset/ --use_intensity False