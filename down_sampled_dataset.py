import sys,os
import time
import math
import numpy as np
import k3d
from scipy.signal import convolve2d
import torch 
import open3d as o3d 
import fpsample 
from utils.data_utils import load_ply
from utils.intensity_correlation_network import ICN


def write_ply(pcd, dir_path, filename):
    ply_header = 'ply\n format ascii 1.0\n comment Exported by visionary python samples\n element vertex 217088\n property float32 x\n property float32 y\n property float32 z\n property uint8 r\n property uint8 g\n property uint8 b\n property float32 i\n end_header\n'
    with open(os.path.join(dir_path, filename), 'w') as f:
        f.write(ply_header)
        for i in range(pcd.shape[0]):
            line = str(pcd[i]).replace('[','')
            line = line.replace(']','')
            f.write(line)
            f.write('\n')

def downsample_pcd(pointcloud, downsample_point):
    fps_indices = fpsample.fps_sampling(pointcloud, downsample_point)
    downsampled_pc = pointcloud[fps_indices][:, :3]
    return downsampled_pc

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                diff_points = min(hist_1_points, hist_2_points) / np.abs(hist_1_points - hist_2_points) + 1e-6
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

def make_downsampled_dataset(data_dir, new_dataset_dir, n_sample):
    os.makedirs(data_dir, exist_ok=True)



