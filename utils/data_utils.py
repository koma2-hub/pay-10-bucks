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
