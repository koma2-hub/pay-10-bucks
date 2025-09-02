import numpy as np


def load_ply(filename):
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
            
            return points
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
    

def l2_norm(a, b):
    return ((a - b) ** 2).sum(axis =1)

def farthest_point_sampling(pcd, k, metrics = l2_norm):
    indices = np.zeros(k, dtype=np.int32)
    distances = np.zeros((k, pcd.shape[0]), dtype=np.float32)
    indices[0] = np.random.randint(len(pcd))
    farthest_point = pcd[indices[0]]
    min_distances = metrics(farthest_point, pcd)
    distances[0, :] = min_distances
    for i in range(1,k):
        indices[i] = np.argmax(min_distances)
        farthest_point = pcd[indices[i]]
        distances[i, :] = metrics(farthest_point, pcd)
        min_distances = np.minimum(min_distances, distances[i, :])
    return indices

def knn(x: np.ndarray, k: int):
    """
    一つの点群データ `x` の各点について、k近傍点のインデックスをNumPyで計算する。

    Args:
        x: 点群データ。形状は (N, C)。
           N: 点の数
           C: 特徴量次元
        k: 探す近傍点の数。

    Returns:
        np.ndarray: 各点のk近傍点のインデックス。形状は (N, k)。
    """
    num_points = x.shape[0]

    # kが点群の総点数を超えないように調整
    k = min(k, num_points)
    if k <= 1:
        k = 1
    x_norm_sq = np.sum(x**2, axis=1, keepdims=True)  # 形状: (N, 1)
    dot_product = np.matmul(x, x.T)  # 形状: (N, N)
    dist_matrix = x_norm_sq - 2 * dot_product + x_norm_sq.T  # 形状: (N, N)
    indices = np.argpartition(dist_matrix, k, axis=1)[:, :k]

    return indices

def create_intensity_histogram(point_cloud_intensity_data, bins=4, density=True):
    """
    輝度値の配列から正規化されたヒストグラムを作成する関数
    """
    # ヒストグラムを作成
    hist, bin_edges = np.histogram(
        point_cloud_intensity_data,
        bins=bins,
        range=(0.0, 1.0), # 輝度値が0.0~1.0に正規化されていると仮定
        density=density  # Trueにするとヒストグラムの面積が1になるように正規化される
    )
    
    # # (別オプション) L2正規化
    # hist = hist.astype(np.float32)
    # l2_norm = np.linalg.norm(hist)
    # if l2_norm > 0:
    #     hist /= l2_norm
        
    return hist
