import os
import sys
import numpy as np
import fpsample


def rotation(pcd, rotation_angle=[np.pi, np.pi, np.pi]):
    # (この関数は変更ありません)
    #ランダムな回転行列の生成
    angle_x = np.random.uniform(rotation_angle[0])
    angle_y = np.random.uniform(rotation_angle[1])
    angle_z = np.random.uniform(rotation_angle[2])

    sinx = np.sin(angle_x)
    cosx = np.cos(angle_x)
    siny = np.sin(angle_y)
    cosy = np.cos(angle_y)
    sinz = np.sin(angle_z)
    cosz = np.cos(angle_z)

    #各軸の回転行列
    rotation_x = np.array([[1, 0,    0],
                           [0, cosx, -sinx],
                           [0, sinx, cosx]])
    rotation_y = np.array([[cosy, 0, siny],
                           [0,    1, 0],
                           [-siny, 0, cosy]])
    rotation_z = np.array([[cosz, -sinz, 0],
                           [sinz, cosz,  0],
                           [0,    0,     1]])
    
    rotation_matrix = rotation_x.dot(rotation_y).dot(rotation_z)

    pcd_rotated = pcd.copy()

    pcd_rotated = pcd_rotated.T
    pcd_rotated[:3, :] = np.dot(rotation_matrix, pcd_rotated[:3, :])
    
    return pcd_rotated.T
    
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
    
def write_ply(pcd, dir_path, filename):
    ply_header = 'ply\n format ascii 1.0\n comment Exported by visionary python samples\n element vertex 217088\n property float32 x\n property float32 y\n property float32 z\n property uint8 r\n property uint8 g\n property uint8 b\n property float32 i\n end_header\n'
    with open(os.path.join(dir_path, filename), 'w') as f:
        f.write(ply_header)
        for i in range(pcd.shape[0]):
            line = str(pcd[i]).replace('[','')
            line = line.replace(']','')
            f.write(line)
            f.write('\n')

def downsample_pcd(pointcloud, downsample_point, intensity=False) -> np.ndarray:
    if pointcloud.shape[0] < downsample_point:
        downsample_point = pointcloud.shape[0]
    fps_indices = fpsample.fps_sampling(pointcloud[:,:3], downsample_point)
    downsampled_pc = pointcloud[fps_indices][:, :4]
    return downsampled_pc