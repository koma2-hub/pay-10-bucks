import sys
import time
import torch
import math
import numpy as np
import open3d as o3d
import k3d
import fpsample
from scipy.signal import convolve2d

# add path to utils
sys.path.append('../')
from utils.grid_histogram_overlap import get_intensity_histogram
from utils.intensity_correlation_network import ICN
from utils.visualize import visualize_pcd, histogram_based_pcd, one_histogram_based_pcd, plot_histograms

# ---- [警告抑制の例: traittypes 由来のものを無視したい場合 (任意) ] ----
# warnings.filterwarnings("ignore", category=UserWarning, message="Given trait value dtype")

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_ply(filename):
    """
    .ply ファイルを読み込み、
    点群(x, y, z, intensity)の NumPy 配列を返す。
    """
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
            
            return points
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {filename}")
        # 必要に応じて sys.exit(1) などで終了するか、Noneを返す
        return None
    except ValueError as e:
        print(f"PLYファイルの読み込みエラー: {e}")
        return None
    except Exception as e:
        print(f"予期せぬエラーが発生しました（load_ply）: {e}")
        return None

def cos_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_model(model_path, input_size):
    """
    モデルをロードする。
    エラー処理を挟んで、ロード失敗時には None を返す。
    """
    try:
        model = ICN(input_size=input_size)
        # デバイスオフロード対応
        device = get_device()
        # map_location を追加して、GPUがなければCPUに落とす
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        print(f"モデルファイルが見つかりません: {model_path}")
        return None
    except RuntimeError as e:
        print(f"モデルのロードに失敗しました: {e}")
        return None
    except Exception as e:
        print(f"予期せぬエラーが発生しました（load_model）: {e}")
        return None

def apply_weights(matrix, threshold=0.9, weight=1.5):
    weighted_matrix = np.copy(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] >= threshold:
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < matrix.shape[0] and 0 <= nj < matrix.shape[1] and (di != 0 or dj != 0):
                            weighted_matrix[ni, nj] = matrix[ni, nj] * weight
    return weighted_matrix

def find_max_kernel(matrix, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    convolved = convolve2d(matrix, kernel, mode='valid')
    max_sum = np.max(convolved)
    max_index = np.unravel_index(np.argmax(convolved), convolved.shape)
    max_indices = [(max_index[0] + ki, max_index[1] + kj)
                   for ki in range(kernel_size)
                   for kj in range(kernel_size)]
    return max_sum, max_indices

def compute_histograms(src_pcd, tgt_pcd, grid_size, threshold, bin_num, overlap):
    """
    src_pcd, tgt_pcd は torch.Tensor で GPU上にあっても構わないが、
    get_intensity_histogram は基本的に NumPy を想定している場合があるので、
    .cpu().numpy() する等の対応が必要かもしれない。
    """
    # GPU -> CPU -> NumPy
    #src_pcd_np = src_pcd.cpu().numpy() if isinstance(src_pcd, torch.Tensor) else src_pcd
    #tgt_pcd_np = tgt_pcd.cpu().numpy() if isinstance(tgt_pcd, torch.Tensor) else tgt_pcd
    
    try:
        src_hist, src_bin, step_x1, step_y1, window_size_x1, window_size_y1, window_coords1 = get_intensity_histogram(
            src_pcd, grid_size, threshold, bin_num, overlap
        )
        tgt_hist, tgt_bin, step_x2, step_y2, window_size_x2, window_size_y2, window_coords2 = get_intensity_histogram(
            tgt_pcd, grid_size, threshold, bin_num, overlap
        )

        # Precompute window points count for each grid cell
        src_window_points = [
            src_pcd[
                (src_pcd[:, 0] >= x_start) & (src_pcd[:, 0] < x_end) &
                (src_pcd[:, 1] >= y_start) & (src_pcd[:, 1] < y_end)
            ].shape[0]
            for x_start, x_end, y_start, y_end in window_coords1
        ]

        tgt_window_points = [
            tgt_pcd[
                (tgt_pcd[:, 0] >= x_start) & (tgt_pcd[:, 0] < x_end) &
                (tgt_pcd[:, 1] >= y_start) & (tgt_pcd[:, 1] < y_end)
            ].shape[0]
            for x_start, x_end, y_start, y_end in window_coords2
        ]

        return (src_hist, tgt_hist, src_bin, tgt_bin,
                window_coords1, window_coords2, src_window_points, tgt_window_points)
    except Exception as e:
        print(f"ヒストグラム計算中にエラーが発生しました: {e}")
        return None, None, None, None, None, None, None, None

def compute_correlations(src_hist, tgt_hist, model, grid_size, border, device, src_window_points, tgt_window_points):
    """
    相関計算を行う。model が None の場合は None を返す。
    """
    if model is None:
        print("モデルがロードされていません。compute_correlationsをスキップします。")
        return None, None, None

    try:
        result_matrix_1 = np.zeros((grid_size, grid_size))
        result_matrix_2 = np.zeros((grid_size, grid_size))
        result_list = []
        used_indices_src_hist = set()
        used_indices_tgt_hist = set()

        for idx, i in enumerate(src_hist):
            if idx in used_indices_src_hist:
                continue

            max_output = float('-inf')
            max_idy = None

            for idy, j in enumerate(tgt_hist):
                if idy in used_indices_tgt_hist:
                    continue

                hist_1_tensor = torch.tensor(i, dtype=torch.float32).unsqueeze(0).to(device)
                hist_2_tensor = torch.tensor(j, dtype=torch.float32).unsqueeze(0).to(device)

                hist_1_points = src_window_points[idx]
                hist_2_points = tgt_window_points[idy]

                # 窓のポイント数の差を考慮して重みをかける
                diff_points = min(hist_1_points, hist_2_points) / (np.abs(hist_1_points - hist_2_points) + 1e-6)
                if diff_points >= 1:
                    diff_points = 1

                with torch.no_grad():
                    output = model(hist_1_tensor, hist_2_tensor)
                    output *= diff_points
                    predicted = (output > border).float()
                    if output > max_output and predicted == 1:
                        max_output = output.item()
                        max_idy = idy

            if max_idy is not None:
                y1, x1 = divmod(idx, grid_size)
                y2, x2 = divmod(max_idy, grid_size)
                result_matrix_1[y1, x1] = max_output
                result_matrix_2[y2, x2] = max_output
                result_list.append((idx, max_idy, max_output))

                used_indices_src_hist.add(idx)
                used_indices_tgt_hist.add(max_idy)

        return result_matrix_1, result_matrix_2, result_list
    except RuntimeError as e:
        print(f"GPUメモリエラー等が発生した可能性があります: {e}")
        return None, None, None
    except Exception as e:
        print(f"予期せぬエラーが発生しました（compute_correlations）: {e}")
        return None, None, None

def process_point_clouds(src_pcd, tgt_pcd, model_path, grid_size, threshold, bin_num, overlap, border):
    """
    メインの処理を行う関数。
    """
    try:
        device = get_device()

        # pcd が None の場合はエラーを返す
        if src_pcd is None or tgt_pcd is None:
            print("ソースまたはターゲットの点群データが存在しないため、処理を中断します。")
            return None, None, None, None, None, None, None, None, None
        
        # pytorch Tensor化 & デバイス移行
        if not isinstance(src_pcd, torch.Tensor):
            src_pcd = torch.from_numpy(src_pcd)
        if not isinstance(tgt_pcd, torch.Tensor):
            tgt_pcd = torch.from_numpy(tgt_pcd)
        src_pcd = src_pcd.to(device)
        tgt_pcd = tgt_pcd.to(device)

        # ヒストグラム計算
        (src_hist, tgt_hist, src_bin, tgt_bin,
         window_coords1, window_coords2, 
         src_window_points, tgt_window_points) = compute_histograms(
            src_pcd, tgt_pcd, grid_size, threshold, bin_num, overlap
        )

        # compute_histogramsが失敗した場合
        if src_hist is None or tgt_hist is None:
            print("ヒストグラム計算に失敗したため、以降の処理を中断します。")
            return None, None, None, None, None, None, None, None, None

        # モデル読み込み
        model = load_model(model_path, src_hist.shape[1])
        if model is None:
            print("モデルのロードに失敗したため、処理を中断します。")
            return None, None, None, None, None, None, None, None, None

        start_time = time.time()

        # 相関計算
        result_matrix_1, result_matrix_2, result_list = compute_correlations(
            src_hist, tgt_hist, model, grid_size, border, device, src_window_points, tgt_window_points
        )
        if result_matrix_1 is None or result_matrix_2 is None:
            print("相関計算に失敗したため、処理を中断します。")
            return None, None, None, None, None, None, None, None, None

        # 加重
        src_weighted_matrix = apply_weights(result_matrix_1)
        tgt_weighted_matrix = apply_weights(result_matrix_2)

        kernel_size = math.ceil(grid_size / 2)
        src_max_sum, src_max_indices = find_max_kernel(src_weighted_matrix, kernel_size)
        tgt_max_sum, tgt_max_indices = find_max_kernel(tgt_weighted_matrix, kernel_size)

        src_max_indices = [y * grid_size + x for y, x in src_max_indices]
        tgt_max_indices = [y * grid_size + x for y, x in tgt_max_indices]

        src_icn = histogram_based_pcd(src_pcd, window_coords1, src_max_indices)
        tgt_icn = histogram_based_pcd(tgt_pcd, window_coords2, tgt_max_indices)
        processing_time = time.time() - start_time

        print(f"Processing time for correlations and downsampling: {processing_time:.3f} seconds")

        return (src_icn, tgt_icn,
                src_hist, tgt_hist, 
                src_bin, tgt_bin, 
                result_list,
                window_coords1, window_coords2)

    except Exception as e:
        print(f"予期せぬエラーが発生しました（process_point_clouds）: {e}")
        return None, None, None, None, None, None, None, None, None

def visualize_and_save_results(src_icn, tgt_icn, result_list, src_pcd, tgt_pcd, 
                               src_hist, tgt_hist, src_bin, tgt_bin, 
                               window_coords1, window_coords2):
    """
    結果の可視化と保存を行う。
    """
    # いずれかが None の場合は可視化・保存をスキップ
    if src_icn is None or tgt_icn is None or result_list is None:
        print("可視化・保存対象データが None のためスキップします。")
        return

    # 可視化部分
    try:
        visualize_pcd(src_icn, tgt_icn)
        for corr_hist in result_list:
            corr_hist = list(corr_hist)
            idx_src = corr_hist[0]
            idx_tgt = corr_hist[1]

            print(corr_hist)
            hist1 = one_histogram_based_pcd(src_pcd, window_coords1, idx_src)
            hist2 = one_histogram_based_pcd(tgt_pcd, window_coords2, idx_tgt)
            print(f"window {idx_src} shape: {hist1.shape}")
            print(f"window {idx_tgt} shape: {hist2.shape}")
            print(f"ICN: {corr_hist[2]:.3f}")
            print(f'Cosine similarity: {cos_similarity(src_hist[idx_src], tgt_hist[idx_tgt]):.3f}')

            visualize_pcd(hist1, hist2)
            plot_histograms(bin_list=src_bin, hist_list=src_hist, idx=idx_src, title='Histogram 1')
            plot_histograms(bin_list=tgt_bin, hist_list=tgt_hist, idx=idx_tgt, title='Histogram 2')

        # 保存
        save_ply('/mnt/c/Users/matsu/SICK/pay-10-bucks/test_data/src_icn.ply', src_icn)
        save_ply('/mnt/c/Users/matsu/SICK/pay-10-bucks/test_data/tgt_icn.ply', tgt_icn)
    except Exception as e:
        print(f"可視化・保存処理中にエラーが発生しました: {e}")

def save_ply(filename, pcd):
    """
    点群データを PLY ファイルとして保存する関数。
    出力フォーマットは以下の通り:
      property float32 x
      property float32 y
      property float32 z
      property uint8 r
      property uint8 g
      property uint8 b
      property float32 i
      
    数値の桁指定:
      - x, y, z: 小数点以下4桁まで
      - r, g, b: 整数 (常に 0)
      - i: 小数点以下2桁まで

    例:
      -1334.0197 -1060.7484 1785.6458 0 0 0 0.16

    入力:
      pcd: (N,3) または (N,4) の numpy 配列または torch.Tensor
           4列目が存在する場合は intensity として使用、なければ 0 とする。
    """
    # Noneチェック
    if pcd is None:
        print(f"保存対象が None のためスキップします: {filename}")
        return

    # torch.Tensor の場合は numpy 配列に変換
    if isinstance(pcd, torch.Tensor):
        pcd_np = pcd.cpu().numpy()
    else:
        pcd_np = np.asarray(pcd)

    # 入力の形状チェック (N,3) または (N,4)
    if pcd_np.ndim != 2 or pcd_np.shape[1] not in (3, 4):
        raise ValueError("pcd は (N,3) または (N,4) の形状である必要があります。")

    # 座標は float32 として取得
    xyz = pcd_np[:, :3].astype(np.float32)
    
    # intensity の取得: 4列目があればその値、なければ 0
    if pcd_np.shape[1] == 4:
        intensity = pcd_np[:, 3].astype(np.float32).reshape(-1, 1)
    else:
        intensity = np.zeros((pcd_np.shape[0], 1), dtype=np.float32)
    
    # r, g, b は uint8 の 0 として生成
    rgb = np.zeros((pcd_np.shape[0], 3), dtype=np.uint8)
    
    # x, y, z, r, g, b, i の順にデータを結合 (shape: (N,7))
    data = np.hstack((xyz, rgb, intensity))

    # PLY ヘッダーの作成
    header = f"""ply
format ascii 1.0
element vertex {data.shape[0]}
property float32 x
property float32 y
property float32 z
property uint8 r
property uint8 g
property uint8 b
property float32 i
end_header
"""
    # ファイルにヘッダーと各点のデータを書き出す
    with open(filename, "w") as f:
        f.write(header)
        for row in data:
            # 書式: x,y,z は小数点以下4桁、i は小数点以下2桁で出力
            f.write(f"{row[0]:.4f} {row[1]:.4f} {row[2]:.4f} {int(row[3])} {int(row[4])} {int(row[5])} {row[6]:.2f}\n")

def downsample_pcd(pointcloud, downsample_point, intensity=False) -> np.ndarray:
    fps_indices = fpsample.fps_sampling(pointcloud, downsample_point)
    downsampled_pc = pointcloud[fps_indices][:, :3]
    if intensity:
        i = pointcloud[fps_indices][:, 3].reshape(-1, 1)
        downsampled_pc = np.concatenate([downsampled_pc, i], axis=1)
    return downsampled_pc

# -----------------------------
# 実際のスクリプト実行部
# -----------------------------
# 例: 
SRC_PATH = '/mnt/c/Users/matsu/SICK/pay-10-bucks/test_data/test_src3.ply'
TGT_PATH = '/mnt/c/Users/matsu/SICK/pay-10-bucks/test_data/test_tgt3.ply'
grid_size = 7
overlap = 0.5
threshold = 100
bin_num = 64
border = 0.6

# ファイル読み込み
src_data = load_ply(SRC_PATH)
tgt_data = load_ply(TGT_PATH)

if src_data is None or tgt_data is None:
    print("点群ファイルが正しく読み込めなかったため、終了します。")
    sys.exit(1)

# モデルパス取得
model_path = '/mnt/c/Users/matsu/SICK/pay-10-bucks/variation5/grid7_bin64/model/correlation_model.pth'


# メイン処理
(src_icn, tgt_icn,
    src_hist, tgt_hist,
    src_bin, tgt_bin,
    result_list,
    window_coords1, window_coords2) = process_point_clouds(
    src_data, tgt_data, model_path, grid_size, threshold, bin_num, overlap, border
)

if src_icn is None:
    print("処理が中断されたため、終了します。")
    sys.exit(1)

# 入力点群の可視化（型変換の警告が出る場合は適切にキャストする）
print("input point cloud:")
try:
    # np.float32に変換して警告を抑制(必要に応じて)
    src_pcd_for_vis = src_data.astype(np.float32) if src_data is not None else None
    tgt_pcd_for_vis = tgt_data.astype(np.float32) if tgt_data is not None else None
    visualize_pcd(src_pcd_for_vis, tgt_pcd_for_vis)
except Exception as e:
    print(f"点群可視化中にエラーが発生しました: {e}")

print("Histograms:")
print(src_hist.shape, tgt_hist.shape)
print("Correlation results from src_hist to tgt_hist:")
print(result_list)
# Print sub point cloud shapes
print('Source ICN downsampled shape:', src_icn.shape)
print('Target ICN downsampled shape:', tgt_icn.shape)

"""
# 可視化 & 保存処理
visualize_and_save_results(
    src_icn, tgt_icn, result_list, 
    torch.from_numpy(src_data),  # visualize用に Tensor化する
    torch.from_numpy(tgt_data), 
    src_hist, tgt_hist, 
    src_bin, tgt_bin, 
    window_coords1, window_coords2
)
"""

save_ply(filename='/mnt/c/Users/matsu/SICK/pay-10-bucks/test_data/test_src3_icn.ply' ,pcd=src_icn)
save_ply(filename='/mnt/c/Users/matsu/SICK/pay-10-bucks/test_data/test_tgt3_icn.ply', pcd=tgt_icn)

src_icn_fps = downsample_pcd(src_icn, 1024, intensity=True)
tgt_icn_fps = downsample_pcd(tgt_icn, 1024, intensity=True)

save_ply(filename='/mnt/c/Users/matsu/SICK/pay-10-bucks/test_data/test_src3_icn_fps.ply', pcd=src_icn_fps)
save_ply(filename='/mnt/c/Users/matsu/SICK/pay-10-bucks/test_data/test_tgt3_icn_fps.ply', pcd=tgt_icn_fps)

