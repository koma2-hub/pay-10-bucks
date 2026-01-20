import numpy as np
from scipy.spatial.transform import Rotation
import random
import open3d as o3d
import sys
import os
import fpsample


def load_ply(filename, intensity=True):
    #.plyファイルを読み込み　点群(x, y, z, intensity)のnumpy配列を返す
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            header_index = None
            for i, line in enumerate(lines):
                if 'end_header' in line:
                    header_index = i
                    break
            if header_index is None:
                raise ValueError("PLYファイルのヘッダが正しく読み込めませんでした。")
            
            # ヘッダ以降の行を読み込み
            points = np.array([list(map(float, l.split())) for l in lines[header_index+1:]])
            if points.shape[1] < 4:
                # xyz + intensity(もしくは他の属性)がなければエラー
                raise ValueError(f"期待される列数に満たないデータが検出されました: {points.shape[1]}列")
            
            # [x, y, z, intensity] の形に整形
            # intensity が最後の列にあると仮定 (points[:, -1])
            points = np.concatenate([points[:, :3], points[:, -1].reshape(-1, 1)], axis=1)
            if(intensity):
                return points
            else:
                return points[:, :3]
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {filename}")
        # 必要に応じて sys.exit(1) などで終了するか、Noneを返す
        return None
    except ValueError as e:
        print(f"PLYファイルの読み込みエラー: {e}")
        return None
    except Exception as e:
        print(f"予期せぬエラーが発生しました(load_ply): {e}")
        return None
    
def downsample_pcd(pointcloud, downsample_point, intensity=False) -> np.ndarray:
    if pointcloud.shape[0] < downsample_point:
        downsample_point = pointcloud.shape[0]
    fps_indices = fpsample.fps_sampling(pointcloud[:,:3], downsample_point)
    downsampled_pc = pointcloud[fps_indices][:, :4]
    return downsampled_pc


def visualize_pcd(pcd_list, window_name = "Point Cloud"):
    # (変更ありません)
    pcd_o3d_list = []
    for pcd in pcd_list:
        pointcloud = pcd[:,:3]
        pcd_obj = o3d.geometry.PointCloud()
        pcd_obj.points = o3d.utility.Vector3dVector(pointcloud)
        rgb = [random.uniform(0,1) for i in range(3)]
        pcd_obj.paint_uniform_color(rgb)
        pcd_o3d_list.append(pcd_obj)
    o3d.visualization.draw_geometries(pcd_o3d_list, window_name=window_name)

def dilate_numpy(img, iterations=1):
    """
    NumPyのみで画像の膨張処理(Dilation)を行う関数。
    近傍(上下左右)の最大値を取ることで白い領域(エッジ)を広げます。
    """
    src = img.copy()
    for _ in range(iterations):
        # パディング（端の処理用）
        padded = np.pad(src, 1, mode='edge')
        
        # 上下左右にずらした配列を用意し、それぞれの位置での最大値を取る
        # これにより、明るい画素が隣接ピクセルに伝播する
        src = np.maximum(src, padded[1:-1, 2:])   # 右にずらしたものとmax
        src = np.maximum(src, padded[1:-1, :-2])  # 左
        src = np.maximum(src, padded[2:, 1:-1])   # 下
        src = np.maximum(src, padded[:-2, 1:-1])  # 上
        
        # 斜め方向も含めたい場合は以下も追加
        src = np.maximum(src, padded[:-2, :-2])   # 左上
        src = np.maximum(src, padded[:-2, 2:])    # 右上
        src = np.maximum(src, padded[2:, :-2])    # 左下
        src = np.maximum(src, padded[2:, 2:])     # 右下
        
    return src


def extract_near_edge_points(ply_file_path, pixel_size=10.0, edge_threshold=0.25, dilation_iter=2):
    """
    エッジ検出を行い、その近傍点(Dilationで拡大)を抽出して保存する。
    
    Args:
        dilation_iter (int): エッジを膨張させる回数。大きいほど広い範囲(近傍)が残ります。
    """
    print(f"処理開始: {ply_file_path}")
    
    # 1. データ読み込み
    data_all = load_ply(ply_file_path, intensity=True)
    if data_all is None: return

    x = data_all[:, 0]
    y = data_all[:, 1]
    intensity = data_all[:, 3]

    # 2. 2Dグリッド化
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    width = x_max - x_min
    height = y_max - y_min
    
    if width <= 0 or height <= 0:
        print("データ範囲が無効です")
        return

    bins_x = int(np.ceil(width / pixel_size)) + 1
    bins_y = int(np.ceil(height / pixel_size)) + 1
    
    print(f"グリッドサイズ: {bins_x} x {bins_y}")
    
    ranges = [[x_min, x_max + pixel_size], [y_min, y_max + pixel_size]]
    
    # ヒストグラム生成
    H_sum, _, _ = np.histogram2d(x, y, bins=[bins_x, bins_y], range=ranges, weights=intensity)
    H_count, _, _ = np.histogram2d(x, y, bins=[bins_x, bins_y], range=ranges)

    with np.errstate(divide='ignore', invalid='ignore'):
        image_raw = np.divide(H_sum, H_count)
        image_raw[H_count == 0] = 0

    image_2d = image_raw.T  # (y, x)
    if image_2d.max() > 0:
        image_2d /= image_2d.max()

    # 3. エッジ検出 (Sobel)
    img_pad = np.pad(image_2d, 1, mode='edge')
    gx = (img_pad[:-2, :-2]*-1) + (img_pad[1:-1, :-2]*-2) + (img_pad[2:, :-2]*-1) + \
         (img_pad[:-2, 2:]*1) + (img_pad[1:-1, 2:]*2) + (img_pad[2:, 2:]*1)
    gy = (img_pad[:-2, :-2]*1) + (img_pad[:-2, 1:-1]*2) + (img_pad[:-2, 2:]*1) + \
         (img_pad[2:, :-2]*-1) + (img_pad[2:, 1:-1]*-2) + (img_pad[2:, 2:]*-1)

    magnitude = np.sqrt(gx**2 + gy**2)
    if magnitude.max() > 0:
        magnitude /= magnitude.max()

    # 4. ★ここが追加ポイント: エッジ画像の膨張(Dilation) ★
    # エッジ強度画像を膨張させて「近傍領域」を作ります
    dilated_magnitude = dilate_numpy(magnitude, iterations=dilation_iter)
    
    print(f"エッジ膨張処理: {dilation_iter}回実行")

    # 5. フィルタリング
    ix = ((x - x_min) / pixel_size).astype(int)
    iy = ((y - y_min) / pixel_size).astype(int)
    ix = np.clip(ix, 0, bins_x - 1)
    iy = np.clip(iy, 0, bins_y - 1)
    
    # 膨張させたマップを使って判定
    point_edge_values = dilated_magnitude[iy, ix]
    mask = point_edge_values > edge_threshold
    
    filtered_data = data_all[mask]
    
    remain_count = len(filtered_data)
    ratio = remain_count/len(data_all)*100
    print(f"抽出結果: {len(data_all)} -> {remain_count} 点 ({ratio:.1f}%)")

    return filtered_data

    # 6. 保存
    if remain_count > 0:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {remain_count}\n")
            f.write("property float32 x\n")
            f.write("property float32 y\n")
            f.write("property float32 z\n")
            f.write("property float32 i\n")
            f.write("end_header\n")
            np.savetxt(f, filtered_data, fmt="%.4f %.4f %.4f %.4f")
        print(f"保存完了: {output_file_path}")
    else:
        print("警告: 点が残りませんでした。閾値を下げるか、pixel_sizeを調整してください。")




