#!/bin/bash
echo "Hello, Ubuntu!"
echo "Train with intensity model!"


python main.py --exp_name dcp_with_intensity_edge_pixel100_threshold2_dialation --emb_nn dgcnnv2 --data_path /mnt/d/SICK/pay-10-bucks/3Dto2D/dataset/pixelsize100_threshold2_dialation --use_intensity
python main.py --exp_name dcp_with_intensity_edge_pixel100_threshold3_dialation --emb_nn dgcnnv2 --data_path /mnt/d/SICK/pay-10-bucks/3Dto2D/dataset/pixelsize100_threshold3_dialation --use_intensity
python main.py --exp_name dcp_with_intensity_edge_pixel100_threshold4_dialation --emb_nn dgcnnv2 --data_path /mnt/d/SICK/pay-10-bucks/3Dto2D/dataset/pixelsize100_threshold4_dialation --use_intensity


python main.py --exp_name dcp_with_intensity_edge_pixel150_threshold2_dialation --emb_nn dgcnnv2 --data_path /mnt/d/SICK/pay-10-bucks/3Dto2D/dataset/pixelsize150_threshold2_dialation --use_intensity
python main.py --exp_name dcp_with_intensity_edge_pixel150_threshold3_dialation --emb_nn dgcnnv2 --data_path /mnt/d/SICK/pay-10-bucks/3Dto2D/dataset/pixelsize150_threshold3_dialation --use_intensity
python main.py --exp_name dcp_with_intensity_edge_pixel150_threshold4_dialation --emb_nn dgcnnv2 --data_path /mnt/d/SICK/pay-10-bucks/3Dto2D/dataset/pixelsize150_threshold4_dialation --use_intensity



python main.py --exp_name dcp_with_intensity_edge_pixel150_threshold5_dialation --emb_nn dgcnnv2 --data_path /mnt/d/SICK/pay-10-bucks/3Dto2D/dataset/pixelsize150_threshold5_dialation --epoch 100 --use_intensity
python main.py --exp_name dcp_with_intensity_edge_pixel100_threshold5_dialation --emb_nn dgcnnv2 --data_path /mnt/d/SICK/pay-10-bucks/3Dto2D/dataset/pixelsize100_threshold5_dialation --epoch 100 --use_intensity


echo "Done!!"






