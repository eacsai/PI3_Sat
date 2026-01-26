import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定使用的GPU设备ID
import sys
sys.path.append('.')

import hydra
import trainers

@hydra.main(version_base="1.2", config_path="../configs", config_name="megadepthsat")
def main(hydra_cfg):
    trainer = eval(hydra_cfg.trainer)(hydra_cfg)
    trainer.train()

if __name__ == '__main__':
    main()