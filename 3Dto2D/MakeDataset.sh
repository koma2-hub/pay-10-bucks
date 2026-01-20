#!/bin/bash
echo "Hello, Ubuntu!"
echo "Make Dataset!!"
python dataset.py --pixel_size 10 --threshold 0.2
python dataset.py --pixel_size 10 --threshold 0.3
python dataset.py --pixel_size 10 --threshold 0.4
python dataset.py --pixel_size 10 --threshold 0.5

python dataset.py --pixel_size 15 --threshold 0.2
python dataset.py --pixel_size 15 --threshold 0.3
python dataset.py --pixel_size 15 --threshold 0.4
python dataset.py --pixel_size 15 --threshold 0.5

