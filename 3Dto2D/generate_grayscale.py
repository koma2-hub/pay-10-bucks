import numpy as np
import random
import sys
import os

# --- 必要な場合は可視化用ライブラリをインポート ---
try:
    import open3d as o3d
    USE_OPEN3D = True
except ImportError:
    USE_OPEN3D = False
    print("Open3Dがインストールされていないため、可視化はスキップされます。")

def load_ply(filename, intensity=True):
    """PLYファイルを読み込み、[x, y, z, i]の形式で返す"""
    if not os.path.exists(filename):
        print(f"ファイルが見つかりません: {filename}")
        return None

    try:
        header_lines = 0
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                header_lines += 1
                if line.strip() == "end_header":
                    break
        
        # データを読み込み
        data = np.loadtxt(filename, skiprows=header_lines)
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        # 列数のチェックと整形
        if data.shape[1] < 3:
            return None
        
        # 3列(xyz)しかない場合はintensityを0で埋める、4列以上なら最後の列を使う
        if data.shape[1] == 3:
            points = np.column_stack((data, np.zeros(data.shape[0])))
        else:
            # xyzと最後の列(intensity)を結合
            points = np.column_stack((data[:, :3], data[:, -1]))
            
        return points[:, :4] if intensity else points[:, :3]

    except Exception as e:
        print(f"PLY読み込みエラー: {e}")
        return None

def visualize_pcd(pcd_list, window_name="Point Cloud"):
    """点群リストを受け取って表示する"""
    if not USE_OPEN3D:
        return

    if not pcd_list or pcd_list[0] is None:
        print("表示するデータがありません。")
        return

    o3d_objects = []
    for pcd in pcd_list:
        if pcd.shape[0] == 0: continue
        
        pointcloud = pcd[:, :3]
        pcd_obj = o3d.geometry.PointCloud()
        pcd_obj.points = o3d.utility.Vector3dVector(pointcloud)
        
        # ランダムな色を設定
        rgb = [random.uniform(0, 1) for _ in range(3)]
        pcd_obj.paint_uniform_color(rgb)
        o3d_objects.append(pcd_obj)
    
    if o3d_objects:
        o3d.visualization.draw_geometries(o3d_objects, window_name=window_name)
    else:
        print("表示可能な点がありません。")

# --- 画像処理関数 (NumPyのみ) ---

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

def extract_near_edge_points(ply_file_path, output_file_path, pixel_size=10.0, edge_threshold=0.25, dilation_iter=2):
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

# --- メイン実行部 ---

input_ply = '/mnt/d/SICK/pay-10-bucks/data/mylabs/processed/robot_record86.ply'
output_ply = 'edge_near_points.ply'

# パラメータ調整
# pixel_size: 画像化の粗さ。大きいほどノイズが減るが大雑把になる。
# dilation_iter: 近傍をどれくらい広げるか。0ならエッジのみ、数字を増やすと太くなる。
extract_near_edge_points(input_ply, output_ply, 
                         pixel_size=10.0, 
                         edge_threshold=0.25, 
                         dilation_iter=2)
pcd = load_ply(input_ply)
# 結果の確認
if USE_OPEN3D and os.path.exists(output_ply):
    result_pcd = load_ply(output_ply)
    if result_pcd is not None:
        # エラー修正箇所: リストに入れて渡す

        visualize_pcd([pcd], window_name="Raw Point Cloud")
        visualize_pcd([result_pcd], window_name="Near Edge Points")