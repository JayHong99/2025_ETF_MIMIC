#!/bin/bash
source ../robust_venv/bin/activate

cfg_path1="configs/MM_mortality_90days_BS_512_hidden_128.yaml"
cfg_path2="configs/MM_readmission_15days_BS_512_hidden_128.yaml"
device=1
fusion_method='WeightedFusion'
# fusion_method='Sum'
# fusion_method='AttnMaskedFusion'

for cfg_path in "$cfg_path1" "$cfg_path2"; do
# for cfg_path in "$cfg_path1"; do
    for seed in 2026 2027 2028; do
        for fusion_method in 'Sum' 'WeightedFusion' 'AttnMaskedFusion'; do
            python mm_run.py --cfg_path "$cfg_path" --device $device --seed $seed --fusion_method "$fusion_method" &
        done
    done
done