import numpy as np
import time
import fpsample 
import open3d as o3d
from utils.data_utils import load_ply , farthest_point_sampling

print('---loading point cloud---')
pcd = load_ply('./test_data/lab_room003.ply')
print('---down sampling point cloud---')
start = time.time()
indices = farthest_point_sampling(pcd, 1024)
sampled_pcd = pcd[indices]
process_time = time.time() - start
print('process time:', process_time)
print('raw point num',pcd.shape[0])
print('sampled point num', sampled_pcd.shape[0])

pcd_o3d = o3d.geometry.PointCloud()
pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:,:3])
pcd_o3d.paint_uniform_color([1,0,0])

start = time.time()
sampled_o3d = pcd_o3d.farthest_point_down_sample(1024)
process_time = time.time() - start
print('process time:', process_time)

sampled_pcd_o3d = o3d.geometry.PointCloud()
sampled_pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:, :3])
sampled_pcd_o3d.paint_uniform_color([0,0,1])

o3d.visualization.draw_geometries([pcd_o3d])


