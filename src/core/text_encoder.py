import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

class TextEncoder(nn.Module):
    def __init__(self, bert_type="emilyalsentzer/Bio_ClinicalBERT", device="cpu"): 
        super().__init__()
        self.bert_type = bert_type # TinyBert "prajjwal1/bert-tiny" in the paper
        self.model = AutoModel.from_pretrained(self.bert_type, output_hidden_states=True)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.device = device
        self.to(device)

    def forward(self, text_tokenized):
        # text_tokenized = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256) # 256 in the paper
        text_tokenized = text_tokenized.to(self.device)
        # check nan
        if torch.isnan(text_tokenized['input_ids']).any():
            raise ValueError("Input contains NaN")
        embeddings = self.model(**text_tokenized).pooler_output
        return embeddings