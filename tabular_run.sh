#!/bin/bash
source ../robust_venv/bin/activate

cfg_path="configs/tabular_mortality_90days_BS_512_hidden_128.yaml"
# cfg_path="configs/tabular_readmission_15days_BS_512_hidden_128.yaml"
device=0

# limit cpu usage ~20%

# for cfg_path in "configs/tabular_mortality_90days_BS_512_hidden_128.yaml" "configs/tabular_readmission_15days_BS_512_hidden_128.yaml"; do
#     for seed in 2026 2027 2028; do
#         python linear_run.py --cfg_path "$cfg_path" --device $device --seed $seed &
#     done
#     wait
# done

# sample
cfg_path="configs/tabular_mortality_90days_BS_512_hidden_128.yaml"
device=1
seed=2026
python linear_run.py --cfg_path "$cfg_path" --device $device --seed $seed --do_train False