import numpy as np
import sys
import os
from utils import load_ply, write_ply, rotation


def remove_outlier(pcd):
    processed = pcd
    index = []
    for i, cord in enumerate(pcd):
        if(abs(cord[0]) > 4000):
            index.append(i)
        elif(abs(cord[1]) > 4000):
            index.append(i)
        elif(abs(cord[2])>4000):
            index.append(i)
    processed = np.delete(pcd, index, axis = 0)
    return processed

            

def main():
    data_folder = "/mnt/c/Users/matsu/SICK/pay-10-bucks/data_surgery/patients"
    data_files = os.listdir(data_folder)
    save_dir = "/mnt/c/Users/matsu/SICK/pay-10-bucks/data/mylabs/processed"

    for file in data_files:
        data_path = os.path.join(data_folder, file)
        pcd = load_ply(data_path)
        angle_xyz = [np.pi, 0, 0]
        rotated_pcd = rotation(pcd, angle_xyz)
        processed_pcd = remove_outlier(rotated_pcd)
        write_ply(processed_pcd, save_dir, file)
        print(file)
        print(processed_pcd.shape)

if __name__ == '__main__':
    main()


