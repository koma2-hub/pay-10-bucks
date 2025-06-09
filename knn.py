import open3d as o3d
import numpy as np

#
def knn(pcd, sample_points = 10):
    assign_index = []
    np_pcd = np.asarray(pcd)
    print(np_pcd)
    for i in range(np_pcd.shape[0]):
        query = i
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        [k, idx, d] = pcd_tree.search_knn_vector_3d(pcd.points[query], sample_points)
        idx = np.asarray(idx)
        for j in range(idx.shape):
            assign_index.append([i, idx])

    return assign_index

def l2_norm(a, b):
    return ((a - b) ** 2).sum(axis = 1)

def farthest_point_sampling(pcd, k, metrics = l2_norm):
    indices = np.zeros(k, dtype=np.int32)
    points = np.asarray(pcd.points)
    distances = np.zeros((k, points.shape[0]), dtype = np.float32)
    indices[0] = np.random.randint(len(points))
    farthest_point = points[indices[0]]
    min_distances = metrics(farthest_point, points)
    distances[0, :] = min_distances
    for i in range(1, k):
        indices[i] = np.argmax(min_distances)
        farthest_point = points[indices[i]]
        distances[i, :] = metrics(farthest_point, points)
        min_distances = np.minimum(min_distances, distances[i, :])
    pcd = pcd.select_by_index(indices)
    return pcd

print("loading src1.ply")
pcd = o3d.io.read_point_cloud("./data/src1.ply")
np_pcd = np.asarray(pcd)
print(np_pcd)


