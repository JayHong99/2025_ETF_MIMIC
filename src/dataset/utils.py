import torch
from transformers import DataCollatorWithPadding, AutoTokenizer

class MIMIC4Collator:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_type)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # debug
        # text_tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        # note_none_token = text_tokenizer("", return_tensors=None, padding=False, truncation=True, max_length=256)
        # sample = text_tokenizer("This is a sample.", return_tensors=None, padding=False, truncation=True, max_length=256)
        # batch = [note_none_token for _ in range(4)] + [sample for _ in range(4)]
        # data_collator = DataCollatorWithPadding(tokenizer=text_tokenizer)
        # batch = data_collator(batch)

    def __call__(self, batch):
        out = {k: [b[k] for b in batch] for k in batch[0]}
        out["age"]       = torch.stack(out["age"])
        out["gender"]    = torch.stack(out["gender"])
        out["ethnicity"] = torch.stack(out["ethnicity"])
        out["types"]     = torch.nn.utils.rnn.pad_sequence(out["types"], batch_first=True, padding_value=0)
        out["codes"]     = torch.nn.utils.rnn.pad_sequence(out["codes"], batch_first=True, padding_value=0)
        out["tabular_flag"] = torch.tensor(out["tabular_flag"])
        
        # 핵심
        out['discharge'] = self.data_collator(out['discharge'])

        out["note_flag"] = torch.tensor(out["note_flag"])
        out["labvectors"] = torch.nn.utils.rnn.pad_sequence(out["labvectors"], batch_first=True, padding_value=0)
        out["lab_flag"]   = torch.tensor(out["lab_flag"])
        out["label"]      = torch.tensor(out["label"])
        return out

class MIMIC4Collator_EMB:
    def __init__(self) : 
        return
    
    def __call__(self, batch) : 
        out = {k : [b[k] for b in batch] for k in batch[0]}
        out["admission_id"] = out["admission_id"]
        out["tab_feature"] = torch.stack(out["tab_feature"])
        out["tab_flag"]    = torch.tensor(out["tab_flag"])
        out["lab_feature"] = torch.stack(out["lab_feature"])
        out["lab_flag"]    = torch.tensor(out["lab_flag"])
        out["note_feature"]= torch.stack(out["note_feature"])
        out["note_flag"]   = torch.tensor(out["note_flag"])
        out["label"]       = torch.stack(out["label"])
        return out