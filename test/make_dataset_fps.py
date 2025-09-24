import os,sys
sys.path.append(os.pardir)
import fpsample
from utils.data_utils import load_ply, save_ply, downsample_pcd
#ダウンサンプリングするデータセットのパスを取得
data_path = "/mnt/c/Users/matsu/SICK/pay-10-bucks/data/mylabs/raw"
file_names = os.listdir(data_path)
save_dir = "/mnt/c/Users/matsu/SICK/pay-10-bucks/data/mylabs_fps/raw"

#データセットを保存するディレクトリの作成
if os.path.isdir(save_dir):
    pass
else:
    os.makedirs(save_dir)


for file in file_names:
    file_path = os.path.join(data_path, file)
    pcd = load_ply(file_path)
    pcd_fps = downsample_pcd(pcd, 1024, intensity=True)
    new_file = "fps1024_" + file
    save_path = os.path.join(save_dir, new_file)
    save_ply(save_path, pcd_fps)
    print("file saved:", save_path)
