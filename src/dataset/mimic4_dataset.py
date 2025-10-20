import os
from pydoc import text
import pandas as pd
import torch
from torch.utils.data import Dataset
from copy import copy
import pickle

# from dataset import tokenizer
from src.dataset.tokenizer import MIMIC4Tokenizer
from src.utils.utils import processed_data_path, read_txt, load_pickle

from transformers import AutoTokenizer

class MIMIC4Dataset(Dataset):
    def __init__(self, config, split, all_hosp_adm_dict, return_raw=False):
        self.config = config
        self.modality = self.config.modality
        self.task = self.config.task
        self.target_days = self.config.target_days
        self.split = split
        self.seed = self.config.seed

        ids_path = processed_data_path / f"mimic4/task:{self.task}_{self.target_days}days/admission_ids_seed_{self.seed}.pkl"
        ids = load_pickle(ids_path)
        ids_key = f"{split}_code_ids" if self.modality == 'tabular' else f"{split}_discharge_ids" if self.modality == 'note' else f"{split}_lab_ids"
        self.included_admission_ids = ids[ids_key]
        print(f"Loaded {len(self.included_admission_ids)} admission ids for {self.split} set.")
        self.return_raw = return_raw
        self.tokenizer = MIMIC4Tokenizer()
        self.all_hosp_adm_dict = all_hosp_adm_dict
        
        self.text_tokenizer = AutoTokenizer.from_pretrained(config.bert_type) #'prajjwal1/bert-tiny'
        self.note_none_token = self.text_tokenizer("", return_tensors=None, padding=False, truncation=True, max_length=256)
        if self.modality == 'note' : 
            self._tokenize()

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
    
class MIMIC4DatasetMM(Dataset):
    def __init__(self, config, split, all_hosp_adm_dict, return_raw=False):
        self.config = config
        self.modality = self.config.modality
        self.task = self.config.task
        self.target_days = self.config.target_days
        self.split = split
        self.seed = self.config.seed

        ids_path = processed_data_path / f"mimic4/task:{self.task}_{self.target_days}days/{split}_admission_ids_seed_{self.seed}.txt"
        ids = read_txt(ids_path)
        self.included_admission_ids = ids
        print(f"Loaded {len(self.included_admission_ids)} admission ids for {self.split} set.")
        self.return_raw = return_raw
        self.tokenizer = MIMIC4Tokenizer()
        self.all_hosp_adm_dict = all_hosp_adm_dict
        
        self.text_tokenizer = AutoTokenizer.from_pretrained(config.bert_type) #'prajjwal1/bert-tiny'
        self.note_none_token = self.text_tokenizer("", return_tensors=None, padding=False, truncation=True, max_length=256)
        self._tokenize()

    def __len__(self):
        return len(self.included_admission_ids)
    
    def _tokenize(self) : 
        for admission_id in self.included_admission_ids :
            hosp_adm = self.all_hosp_adm_dict[admission_id]
            discharge = hosp_adm.discharge
            if discharge is not None : 
                discharge_tokenized = self.text_tokenizer(discharge, return_tensors=None, padding=False, truncation=True, max_length=256)
            else : 
                discharge_tokenized = self.note_none_token
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
        
        age, gender, ethnicity, types, codes = self.load_code(hosp_adm)
        return_dict["age"] = age
        return_dict["gender"] = gender
        return_dict["ethnicity"] = ethnicity
        return_dict["types"] = types
        return_dict["codes"] = codes
        tabular_flag = 1
        
        
        discharge_token = hosp_adm.discharge_tokenized
        return_dict["discharge"] = discharge_token
        note_flag = 1 if hosp_adm.discharge is not None else 0

            
        labvectors = hosp_adm.labvectors
        if labvectors is not None : 
            return_dict["labvectors"] = torch.FloatTensor(labvectors)
            lab_flag = 1
        else : 
            return_dict["labvectors"] = torch.zeros(1, 114)
            lab_flag = 0
            
        label = float(getattr(hosp_adm, self.task))
        label = torch.tensor(label, dtype=torch.long)

        return_dict["tabular_flag"] = tabular_flag
        return_dict["note_flag"] = note_flag
        return_dict["lab_flag"] = lab_flag
        return_dict["label"] = label
        return return_dict