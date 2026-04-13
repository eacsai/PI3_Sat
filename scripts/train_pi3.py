import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # 指定使用的GPU设备ID
import sys
sys.path.append('.')

import hydra
import trainers

@hydra.main(version_base="1.2", config_path="../configs", config_name="googlestreet_sat")
def main(hydra_cfg):
    trainer = eval(hydra_cfg.trainer)(hydra_cfg)
    trainer.train()

if __name__ == '__main__':
    main()

# accelerate launch --config_file configs/accelerate/ddp.yaml --num_processes 7 --num_machines 1 scripts/train_pi3.py