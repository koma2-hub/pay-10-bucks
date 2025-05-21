import os 
import glob
import numpy as np
import torch
from tqdm import tqdm

def get_intensity_histogram(pointcloud, grid_size, threshold = 100, bin_num = 64, overlap = 0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    max_x, max_y , _ = torch.max(pointcloud[:, :3], axis = 0)[0]
    min_x, min_y, _ = torch.min(pointcloud[:, :3], axis = 0)[0]

    max_intensity = torch.max(pointcloud[:, -1])
    min_intensity = torch.max(pointcloud[:, -1])

    max_x, max_y = float(max_x), float(max_y)
    min_x, min_y = float(min_x), float(min_y)
    '''
    上でキャストしなくても
    window_size_x = float((max_x - min_x) / (grid_size - (grid_size - 1)* overlap))
    下でとかでもいいのでは？
    '''
    window_size_x = (max_x - min_x) / (grid_size - (grid_size - 1) * overlap)
    window_size_y = (max_y - min_y) / (grid_size - (grid_size - 1) * overlap)

    step_x = window_size_x * (1 - overlap)
    step_y = window_size_y * (1 - overlap)

    intensity_list = []
    hist_list = []
    bin_list = []
    index_list = []
    point_num_list = []
    all_points = []

    window_coords = []

    for i in range(grid_size):
        for j in range(grid_size):
            x_start = min_x + j * step_x
            x_end = x_start + window_size_x
            y_start = min_y + i * step_y
            y_end = y_start + window_size_y

            window_coords.append((x_start, x_end, y_start, y_end))

            sample_pcd = pointcloud[
                (pointcloud[:, 0] >= x_start) & (pointcloud[:, 0] < x_end) &
                (pointcloud[:, 1] >= y_start) & (pointcloud[:, 1] < y_end)
            ]
            if sample_pcd.shape[0] <= threshold:
                continue
            all_points += sample_pcd.shape[0]

            point_num_list.append(sample_pcd.shape[0])
            intensity = sample_pcd[:, -1]
            intensity_list.append(intensity)
            #ここの輝度値の最小最大ってWindowの奴じゃなくてデータ全体の最大最小値だけどいいの？
            hist = torch.histc(intensity, bins = bin_num, min = min_intensity, max = max_intensity)
            hist = hist / torch.sum(hist) #normalize
            #最大値から最小値までｂｉｎに基づいて等差数列を作る
            bins = torch.linspace(min_intensity, max_intensity, bin_num).to(device)
            hist = hist.detach().cpu().numpy()
            bins = bins.detach().cpu().numpy()

            hist_list.append(hist)
            bin_list.append(bins)
            index_list.append([j, i])

    return np.array(hist_list), np.array(bin_list), step_x, step_y, window_size_x, window_size_y, window_coords



