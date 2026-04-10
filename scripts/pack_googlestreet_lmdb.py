import os
import sys
import lmdb
import json
from tqdm import tqdm

sys.path.append('.')
from datasets.googlestreet_dataset import get_sorted_pair_paths

def pack_to_lmdb(data_root='/data/zhongyao/dataset', lmdb_path='/data/wangqw/dataset_lmdb'):
    print(f"Packing {data_root} into LMDB at {lmdb_path}")
    if not os.path.exists(data_root):
        print(f"Directory {data_root} does not exist.")
        return

    # 收集所有的 pair 路径 (混合 train 和 test)
    pair_paths_train = get_sorted_pair_paths(data_root, split=False, mode='train')
    pair_paths_test = get_sorted_pair_paths(data_root, split=False, mode='test')
    pair_paths = list(set(pair_paths_train + pair_paths_test))
    
    # 为了防止多次运行导致 LMDB 空间膨胀和脏数据积累，我们在重新打包前先自动清理旧环境
    if os.path.exists(lmdb_path):
        import shutil
        print(f"Cleaning ancient LMDB at {lmdb_path}...")
        shutil.rmtree(lmdb_path)
    os.makedirs(lmdb_path, exist_ok=True)

    # map_size 设为 100TB 作为理论天花板 (采用内存映射按需分配，并非实际占用)
    # 因为 64位系统虚拟内存极大，我们只需保证该数值远远超过实际数据集体积即可避免 MapFull 崩溃
    map_size = 5 * 1099511627776 
    env = lmdb.open(lmdb_path, map_size=map_size, writemap=True)
    
    dir_cache = {}
    
    with env.begin(write=True) as txn:
        for folder_path in tqdm(pair_paths, desc="Packing pairs into LMDB"):
            if not os.path.exists(folder_path):
                continue
            
            files = os.listdir(folder_path)
            npy_configs = [f for f in files if f.endswith('_rgb.npy')]
            dir_cache[folder_path] = npy_configs
            
            # 明确打包 .npy / .pt / .jpg 等所有数据集使用到的文件
            files_to_pack = [
                f for f in files 
                if f.endswith('_rgb.npy') or f.endswith('_depth.npy') or f.endswith('_depth_dap.pt') or f.endswith('.jpg')
            ]
            
            for fname in files_to_pack:
                file_path = os.path.join(folder_path, fname)
                # 读取原文件的纯二进制内容
                if os.path.isfile(file_path):
                    with open(file_path, 'rb') as f:
                        content = f.read()
                    # 使用 file_path 作为 key
                    txn.put(file_path.encode('utf-8'), content)
                
    # 将目录列表缓存存入 LMDB
    with env.begin(write=True) as txn:
        txn.put(b'__DIR_CACHE__', json.dumps(dir_cache).encode('utf-8'))
        
    env.close()
    print(f"Done! LMDB saved at {lmdb_path}")

if __name__ == '__main__':
    pack_to_lmdb()
