from torch_geometric.nn import knn
from utils.data_utils import load_ply
from DGCNN import EdgeConv
import torch

src = load_ply('./data/src1.ply')
trg = load_ply('./data/trg1.ply')
print(src.shape)
print(trg.shape)

src = torch.from_numpy(src)
trg = torch.from_numpy(trg)
assign_index_src = knn(x = src[:, -1], y = src[:, -1], k = 20, batch_x = src, batch_y = src)
print("assign_index src:", assign_index_src.shape)
assign_index_trg = knn(x = trg[:, -1], y = trg[:, -1], k = 20, batch_x = trg, batch_y = trg)
print("assign index trg:", assign_index_trg)


