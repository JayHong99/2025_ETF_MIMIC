#!/bin/bash
source ../robust_venv/bin/activate

cfg_path="configs/note_mortality_90days_BS_512_hidden_128.yaml"
# cfg_path="configs/note_readmission_15days_BS_512_hidden_128.yaml"
device=0

max_jobs=2


cfg_path1="configs/note_mortality_90days_BS_512_hidden_128.yaml"
cfg_path2="configs/note_readmission_15days_BS_512_hidden_128.yaml"

# for cfg_path in "$cfg_path1" "$cfg_path2"; do
#     for seed in 2026 2027 2028 ; do
#         python linear_run.py --cfg_path "$cfg_path" --device $device --seed $seed &
        
#         while [ "$(jobs -r | wc -l)" -ge "$max_jobs" ]; do
#             wait -n
#         done
#     done
# done

seed=2026
python linear_run.py --cfg_path "$cfg_path1" --device $device --seed $seed