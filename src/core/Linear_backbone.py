import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.core.tabular_encoder import TabularEncoder 
from src.core.text_encoder import TextEncoder
from src.core.rnn_encoder import RNNEncoder


                
class LinearBackbone(nn.Module) : 
    def __init__(self, config, tokenizer) : 
        super(LinearBackbone, self).__init__()
        self.config = config
        self.modality = config.modality
        self.training=True
        
        if self.modality == 'tabular' : 
            self.encoder = TabularEncoder(config, tokenizer)
            self.projection_layer = nn.Linear(config.embedding_size, config.projection_size) # embedding size is tabular embedding size
        elif self.modality == 'note' :
            self.encoder = TextEncoder(bert_type=config.bert_type, device='cuda')
            output_dim = self.encoder.model.config.hidden_size
            print("Fine-tuning BERT model")
            self.projection_layer = nn.Linear(output_dim, config.projection_size) # 128 to 512
        elif self.modality == 'lab' : 
            self.encoder = RNNEncoder(input_size=114,
                                      hidden_size=config.embedding_size,
                                      num_layers=config.rnn_layers,
                                      rnn_type=config.rnn_type,
                                      dropout=config.dropout,
                                      bidirectional=config.rnn_bidirectional,
                                      device='cuda')
            self.projection_layer = nn.Linear(config.embedding_size, config.projection_size)
            
        else:
            raise NotImplementedError
        
        self.classifier = Proto_Classifier(feat_in=config.projection_size, num_classes=2)
        
    def forward(
            self,
            age,
            gender,
            ethnicity,
            types,
            codes,
            tabular_flag,
            discharge,
            note_flag,
            labvectors,
            lab_flag,
            # label,
            **kwargs
    ):
        if self.modality == 'tabular' :             
            embedding =  self.encoder(codes, types, age, gender, ethnicity)
        elif self.modality == 'note' :
            embedding = self.encoder(discharge)
        elif self.modality == 'lab' :
            embedding = self.encoder(labvectors)
        else:
            raise NotImplementedError

        
        probs = self.projection_layer(embedding)
        return probs
    
class Proto_Classifier(nn.Module):
    def __init__(self, feat_in, num_classes):
        super(Proto_Classifier, self).__init__()
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(num_classes / (num_classes-1)) * torch.matmul(P, I-((1/num_classes) * one))
        self.register_buffer('proto', M)

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_classes), atol=1e-06), torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
        return P

    def load_proto(self, proto):
        print(f"ETF vecter Loaded")
        # self.proto = copy.deepcopy(proto)
        self.proto.data.copy_(proto.to(self.proto.device))

    def forward(self, label):
        # produce the prototypes w.r.t. the labels
        target = self.proto[:, label].T ## B, d  output: B, d
        return target