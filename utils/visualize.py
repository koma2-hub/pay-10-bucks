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