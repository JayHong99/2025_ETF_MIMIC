import numpy as np
import torch
from torch.utils.data import Dataset

from tqdm import tqdm
from src.utils.utils import processed_data_path, read_txt, load_pickle


class MIMIC4Dataset_EMB(Dataset):
    def __init__(self, config, split, return_raw=False):
        self.config = config
        self.modality = self.config.modality
        self.task = self.config.task
        self.target_days = self.config.target_days
        self.split = split
        self.seed = self.config.seed

        tab = load_pickle(self.config.tabular_run_dir / 'outputs' / 'best_epoch.pkl')[split]['outputs']
        lab = load_pickle(self.config.lab_run_dir / 'outputs' / 'best_epoch.pkl')[split]['outputs']
        note = load_pickle(self.config.note_run_dir / 'outputs' / 'best_epoch.pkl')[split]['outputs']
        
        tab_keys = list(tab.keys())
        lab_keys = list(lab.keys())
        note_keys = list(note.keys())
        self.included_admission_ids = np.array(
            list(set(tab_keys) | set(lab_keys) | set(note_keys))
        )
        print(f"Loaded {len(self.included_admission_ids)} admission ids for {self.split} set.")
        
        # output value : label feature prob loss
        
        new_adm_dict = dict() # {key : [tab_feature, tab_flag, lab_feature, lab_flag, note_feature, note_flag, label]}
        for admission_id in tqdm(self.included_admission_ids, desc=f"Loading {split} set embeddings", ncols=100, leave=True) :
            tab_adm = tab.get(admission_id) # Raise error if not found. Tabular should always exist.
            lab_adm = lab.get(admission_id, None)
            note_adm = note.get(admission_id, None)
            
            tab_feature= tab_adm['features']
            tab_flag = 1 if tab_adm is not None else 0
            lab_feature = lab_adm['features'] if lab_adm is not None else np.zeros(self.config.projection_size)
            lab_flag = 1 if lab_adm is not None else 0
            note_feature = note_adm['features'] if note_adm is not None else np.zeros(self.config.projection_size)
            note_flag = 1 if note_adm is not None else 0
            label = tab_adm['labels']
            
            if tab_flag : 
                tab_feature = tab_feature / np.linalg.norm(tab_feature)
            if lab_flag : 
                lab_feature = lab_feature / np.linalg.norm(lab_feature)
            if note_flag : 
                note_feature = note_feature / np.linalg.norm(note_feature)
            
            new_adm_dict[admission_id] = [
                tab_feature, tab_flag,
                lab_feature, lab_flag,
                note_feature, note_flag,
                label
            ]
        self.all_adm_dict = new_adm_dict
        self.return_raw = return_raw
        
        # memory efficient
        del tab, lab, note, tab_adm, lab_adm, note_adm, tab_keys, lab_keys, note_keys, new_adm_dict 
    
    def __len__(self):
        return len(self.included_admission_ids)
    
    def load_emb(self, admission_id) : 
        adm_data = self.all_adm_dict[admission_id]
        tab_feature = adm_data[0]
        tab_flag = adm_data[1]
        lab_feature = adm_data[2]
        lab_flag = adm_data[3]
        note_feature = adm_data[4]
        note_flag = adm_data[5]
        label = adm_data[6]
        
        return tab_feature, tab_flag, lab_feature, lab_flag, note_feature, note_flag, label
    
    def __getitem__(self, index):
        admission_id = str(self.included_admission_ids[index])
        tab_feature, tab_flag, lab_feature, lab_flag, note_feature, note_flag, label = self.load_emb(admission_id)
        
        return_dict = dict()
        return_dict["admission_id"] = torch.LongTensor([int(admission_id)])
        return_dict["tab_feature"] = torch.FloatTensor(tab_feature)
        return_dict["tab_flag"] = torch.LongTensor([tab_flag])
        return_dict["lab_feature"] = torch.FloatTensor(lab_feature)
        return_dict["lab_flag"] = torch.LongTensor([lab_flag])
        return_dict["note_feature"] = torch.FloatTensor(note_feature)
        return_dict["note_flag"] = torch.LongTensor([note_flag])
        return_dict["label"] = torch.LongTensor([label])
        return return_dict