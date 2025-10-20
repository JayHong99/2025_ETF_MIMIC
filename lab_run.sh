#!/bin/bash
source ../robust_venv/bin/activate
device=0

# cfg_path="configs/lab_mortality_90days_BS_512_hidden_128.yaml"

# for seed in 2026 2027 2028; do
#     cpulimit -l 400 -- python linear_run.py --cfg_path "$cfg_path" --device $device --seed $seed
#     wait
# done


cfg_path="configs/lab_mortality_90days_BS_512_hidden_128.yaml"
cpulimit -l 400 --f -- python linear_run.py --cfg_path "$cfg_path" --device $device --seed 2026
wait
cpulimit -l 400 --f -- python linear_run.py --cfg_path "$cfg_path" --device $device --seed 2027
wait
cpulimit -l 400 --f -- python linear_run.py --cfg_path "$cfg_path" --device $device --seed 2028
wait


cfg_path="configs/lab_readmission_15days_BS_512_hidden_128.yaml"
cpulimit -l 400 --f -- python linear_run.py --cfg_path "$cfg_path" --device $device --seed 2026
wait
cpulimit -l 400 --f -- python linear_run.py --cfg_path "$cfg_path" --device $device --seed 2027
wait
cpulimit -l 400 --f -- python linear_run.py --cfg_path "$cfg_path" --device $device --seed 2028
wait

echo "All tasks completed."