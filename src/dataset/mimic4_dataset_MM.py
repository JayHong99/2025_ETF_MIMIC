import os
from pydoc import text
import pandas as pd
import torch
from torch.utils.data import Dataset
from copy import copy

# from dataset import tokenizer
from src.dataset.tokenizer import MIMIC4Tokenizer
from src.utils.utils import processed_data_path, read_txt, load_pickle

from transformers import AutoTokenizer

class MIMIC4Dataset_EMB(Dataset):
    def __init__(self, split, task, seed, load_no_label=False, dev=False, return_raw=False):
        if dev:
            assert split == "train"
        if load_no_label:
            assert split == "train"
        self.load_no_label = load_no_label
        self.split = split
        self.task = task
        if self.task == "mortality":
            hosp_adm_dict_path = os.path.join(processed_data_path, "mimic4/hosp_adm_dict_90days.pkl")
            included_admission_ids_path = os.path.join(processed_data_path, f"mimic4/task:{task}_90days/{split}_admission_ids_seed_{seed}.txt")
        elif self.task == "readmission" : 
            hosp_adm_dict_path = os.path.join(processed_data_path, "mimic4/hosp_adm_dict_15days.pkl")
            included_admission_ids_path = os.path.join(processed_data_path, f"mimic4/task:{task}_15days/{split}_admission_ids_seed_{seed}.txt")
            
            
        self.all_hosp_adm_dict = load_pickle(hosp_adm_dict_path)
        included_admission_ids = read_txt(included_admission_ids_path)
        self.no_label_admission_ids = []
        if load_no_label:
            no_label_admission_ids = read_txt(
                os.path.join(processed_data_path, f"mimic4/task:{task}/no_label_admission_ids.txt"))
            self.no_label_admission_ids = no_label_admission_ids
            included_admission_ids += no_label_admission_ids
        self.included_admission_ids = included_admission_ids
        if dev:
            self.included_admission_ids = self.included_admission_ids[:10000]
        self.return_raw = return_raw
        self.tokenizer = MIMIC4Tokenizer()

    def __len__(self):
        return len(self.included_admission_ids)
    
    def _tokenize(self) : 
        for admission_id in self.included_admission_ids :
            hosp_adm = self.all_hosp_adm_dict[admission_id]
            discharge = hosp_adm.discharge
            discharge_tokenized = self.text_tokenizer(discharge, return_tensors=None, padding=False, truncation=True, max_length=256)
            hosp_adm.discharge_tokenized = discharge_tokenized
            self.all_hosp_adm_dict[admission_id] = hosp_adm

    def load_code(self, hosp_adm) : 
        age = str(hosp_adm.age)
        gender = hosp_adm.gender
        ethnicity = hosp_adm.ethnicity
        types = hosp_adm.trajectory[0]
        codes = hosp_adm.trajectory[1]
        if not self.return_raw :
            age, gender, ethnicity, types, codes = self.tokenizer(
                age, gender, ethnicity, types, codes
            )
        return age, gender, ethnicity, types, codes

    def __getitem__(self, index):
        admission_id = str(self.included_admission_ids[index])
        
        return_dict = dict()
        return_dict["id"] = admission_id
        
        hosp_adm = self.all_hosp_adm_dict[admission_id]
        
        tabular_flag, note_flag, lab_flag = True, False, False
        age, gender, ethnicity, types, codes = self.load_code(hosp_adm)
        return_dict["age"] = age
        return_dict["gender"] = gender
        return_dict["ethnicity"] = ethnicity
        return_dict["types"] = types
        return_dict["codes"] = codes
        
        if self.modality == 'note' : 
            tabular_flag, note_flag, lab_flag = False, True, False
            discharge = hosp_adm.discharge_tokenized
            return_dict["discharge"] = discharge

            
        elif self.modality == 'lab' : 
            tabular_flag, note_flag, lab_flag = False, False, True
            labvectors = hosp_adm.labvectors
            return_dict["labvectors"] = torch.FloatTensor(labvectors)
            
            
        if self.modality != "note" : # for tabular and lab
            return_dict["discharge"] = self.note_none_token
        if self.modality != "lab" : # for tabular and note
            return_dict["labvectors"] = torch.zeros(1, 114)

        label = float(getattr(hosp_adm, self.task))
        label = torch.tensor(label, dtype=torch.long)

        return_dict["tabular_flag"] = tabular_flag
        return_dict["note_flag"] = note_flag
        return_dict["lab_flag"] = lab_flag
        return_dict["label"] = label
        return return_dict