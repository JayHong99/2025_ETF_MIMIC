#!/bin/bash
source ../robust_venv/bin/activate

cfg_path1="configs/tabular_mortality_90days_BS_512_hidden_128.yaml"
cfg_path1="configs/tabular_readmission_15days_BS_512_hidden_128.yaml"
device=1


for cfg_path in "$cfg_path1" "$cfg_path2"; do
    for seed in 2026 2027 2028; do
        python linear_run.py --cfg_path "$cfg_path" --device $device --seed $seed &
    done
    wait
done