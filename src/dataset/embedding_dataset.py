import os

import torch
from torch.utils.data import Dataset

from src.utils.utils import load_pickle, processed_data_path


class EMBED_Dataset(Dataset):
    def __init__(self, config, split, pretrained_model_epoch):
        self.config = config
        self.split = split
        self.pretrained_model_epoch = pretrained_model_epoch
        self.task = config.task
        self.target_days = int(config.target_days)

        data_path = self.config.ssl_output_dir / f'{split}_embeddings_epoch{pretrained_model_epoch:03d}.pt'
        self.data = torch.load(data_path)
        self.included_admission_ids = list(self.data.keys())
        
        # /home/data/2025_MIMICIV_processed/mimic4/task:{task}_{target_days}days/noisy_{noisy_rate:.1f}_key_labels.pkl
        
        key_labels = load_pickle(processed_data_path / f"mimic4/task:{self.task}_{self.target_days}days/noisy_{config.linear_eval_dict.noisy_ratio:.1f}_key_labels.pkl")
        self.key_label = key_labels.get(self.config.modality)


    def __len__(self):
        return len(self.included_admission_ids)

    def __getitem__(self, index):
        admission_id = self.included_admission_ids[index]
        data = self.data[admission_id]
        output_dict = dict()
        output_dict['admission_id'] = admission_id
        output_dict['projection'] = torch.tensor(data['projection'], dtype=torch.float32)
        output_dict['original_label'] = torch.tensor(data['labels'], dtype=torch.long)
        output_dict['label' ] = torch.tensor(self.key_label[int(admission_id)], dtype=torch.long)
        return output_dict
