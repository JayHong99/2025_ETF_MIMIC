from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import torch

from joblib import Parallel, delayed



result_dir = Path('/home/jayhong7200/2025_MM_ETF_V2/MIMIC/Results/Linear_MultiModal')

epoch_paths = list(result_dir.rglob('*.pkl'))
epoch_paths = [path for path in epoch_paths if '_score' not in str(path) and 'best_epoch' not in str(path) and 'mortality' in str(path)]

def process_single_path(epoch_path) :
    output = pd.read_pickle(epoch_path)
    for phase in ['train', 'valid', 'test'] : 
        phase_output = output.get(phase)
        phase_outputs = phase_output.get('outputs')
        new_outputs = dict()
        keys = list(phase_outputs.keys())
        print(f"  {phase} num samples: {len(keys)}")
        for key in keys :
            if type(key) == torch.Tensor :
                new_outputs[key.item()] = phase_outputs[key]
            else :
                new_outputs[key] = phase_outputs[key]
        phase_output['outputs'] = new_outputs
    # Save back
    with open(epoch_path, 'wb') as f :
        pickle.dump(output, f)
    print(f"  Saved fixed output to {epoch_path}")
    print('==============================='*5)

# i = 0
# for epoch_path in epoch_paths :
#     process_single_path(epoch_path)

Parallel(n_jobs=16)(delayed(process_single_path)(epoch_path) for epoch_path in epoch_paths)