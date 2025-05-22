import k3d.helpers
import numpy as np
import matplotlib.pyplot as plt
import k3d

def visualize_pcd(pcd1, pcd2):
    if pcd1.shape[1] == 4:
        color_map = k3d.colormaps.matplotlib_color_maps.Greys
        plot = k3d.plot(grid_visible=False)

        colors = k3d.helpers.map_colors(pcd1[:, -1], color_map)
        plt_points = k3d.points(pcd1[:, :3] - [3000, 0, 0], point_size=3, colors=colors)
        plot += plt_points
        
        colors = k3d.helpers.map_colors(pcd2[:, -1], color_map)
        plt_points = k3d.points(pcd2[:, :3], point_size=3, colors=colors)
        plot += plt_points
        
        plot.display()
    else:
        plot = k3d.plot(grid_visible=False)
        plt_points = k3d.points(pcd1 - [3000, 0, 0], point_size=3, color=0xff0000)
        plot += plt_points
        
        plt_points = k3d.points(pcd2, point_size=3, color=0x0000ff)
        plot += plt_points
        
        plot.display()

def histogram_based_pcd(pcd, window_coords, indices):
    sub_pcd = np.empty((0,4))
    pcd = pcd.cup().detach().numpy()

    for idx in indices:
        x_start, x_end, y_start, y_end = window_coords[idx]

        tmp_pcd = pcd[
            (pcd[:, 0] >= x_start) & (pcd[:, 0] < x_end) &
            (pcd[:, 1] >= y_start) & (pcd[:, 1] < y_end)
        ]

        # デバッグ用に抽出されたポイント数を表示
        #print(f"Points in window {idx}: {tmp_pcd.shape[0]}")

        sub_pcd = np.concatenate((sub_pcd, tmp_pcd), axis=0)
    
    return np.unique(sub_pcd, axis=0)

def one_histogram_based_pcd(pcd, window_coords, index):
    x_start, x_end, y_start, y_end = window_coords[index]
    pcd = pcd.cpu().detach().numpy()
    sub_pcd = pcd[
        (pcd[:, 0] >= x_start) & (pcd[:, 0] < x_end) &
        (pcd[:, 1] >= y_start) & (pcd[:, 1] < y_end)
    ]
    return np.unique(sub_pcd, axis=0)

def plot_histograms(bin_list, hist_list, idx, title):
    plt.figure()
    plt.plot(bin_list[idx], hist_list[idx])
    plt.xlabel('Intensity', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title(title, fontsize=16)
    plt.show()

