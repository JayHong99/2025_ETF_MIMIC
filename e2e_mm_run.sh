#!/bin/bash
source ../robust_venv/bin/activate

cfg_path1="configs/E2E_MM_mortality_90days_BS_512_hidden_128.yaml"
cfg_path2="configs/E2E_MM_readmission_15days_BS_512_hidden_128.yaml"
device=1

for cfg_path in "$cfg_path1" "$cfg_path2"; do
    for fusion_method in 'Sum' 'WeightedFusion' 'AttnMaskedFusion'; do
        for seed in 2026 2027 2028; do
            python mm_end_to_end_run.py --cfg_path "$cfg_path" --device $device --seed $seed --fusion_method "$fusion_method"
        done
        wait
    done
done

# cfg_path="$cfg_path1"
# device=0
# seed=2026
# fusion_method='WeightedFusion'
# python mm_end_to_end_run.py --cfg_path "$cfg_path" --device $device --seed $seed --fusion_method "$fusion_method"