import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


                
class MMBackbone(nn.Module) : 
    def __init__(self, config) : 
        super(MMBackbone, self).__init__()
        self.config = config
        self.training=True
                
        self.classifier = Proto_Classifier(feat_in=config.projection_size, num_classes=2)
        print(f"Loading ETF Matrix from {config.etf_path}")
        self.classifier.load_proto(torch.load(config.etf_path))
        
        if config.fusion_method == 'Sum' : 
            self.fusion_layer = SumFusion(self.classifier, input_dim=config.projection_size)
        elif config.fusion_method == 'WeightedFusion' : 
            self.fusion_layer = WeightedFusion(self.classifier)
        elif config.fusion_method == 'AttnMaskedFusion' : 
            self.fusion_layer = AttnMaskedFusion(self.classifier, input_dim=config.projection_size)
        else : 
            raise NotImplementedError(f"Fusion method not implemented. : {config.fusion_method}")
        
    def forward(
            self,
            admission_id,
            tab_feature,
            tab_flag,
            lab_feature,
            lab_flag,
            note_feature,
            note_flag,
            label,
    ):
        # send all to device
        tab_feature = tab_feature.to(self.device, non_blocking=True)
        tab_flag = tab_flag.to(self.device, non_blocking=True)
        lab_feature = lab_feature.to(self.device, non_blocking=True)    
        lab_flag = lab_flag.to(self.device, non_blocking=True)
        note_feature = note_feature.to(self.device, non_blocking=True)
        note_flag = note_flag.to(self.device, non_blocking=True)

        logits = self.fusion_layer(
            tab_feature, tab_flag,
            lab_feature, lab_flag,
            note_feature, note_flag
        )
        return logits
    
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
        self.proto.data.copy_(proto.to(self.proto.device))

    def forward(self, label):
        # produce the prototypes w.r.t. the labels
        target = self.proto[:, label].T ## B, d  output: B, d
        return target
    


class SumFusion(nn.Module):
    def __init__(self, classifier, input_dim):
        super(SumFusion, self).__init__()
        self.projection = nn.Linear(input_dim, input_dim)
        self.classifier = classifier
        

    def forward(self, x, x_flag,
                y, y_flag,
                z, z_flag):
        
        x = x * x_flag.unsqueeze(1) # flag 0 means None -> multiply 0
        y = y * y_flag.unsqueeze(1)
        z = z * z_flag.unsqueeze(1)
        
        feature_sum = x + y + z
        output = self.projection(feature_sum)
        output = torch.nn.functional.normalize(output, p=2, dim=1)
        logit= output @ self.classifier.proto
        return logit


class WeightedFusion(nn.Module):
    def __init__(self, classifier):
        super(WeightedFusion, self).__init__()
        self.weight_x = nn.Parameter(torch.randn(1))
        self.weight_y = nn.Parameter(torch.randn(1))
        self.weight_z = nn.Parameter(torch.randn(1))
        self.classifier = classifier

    def forward(self, x, x_flag,
                y, y_flag,
                z, z_flag):
        
        x = x * x_flag.unsqueeze(1) # flag 0 means None -> multiply 0
        y = y * y_flag.unsqueeze(1)
        z = z * z_flag.unsqueeze(1)
        
        output = self.weight_x * x + self.weight_y * y + self.weight_z * z
        output = torch.nn.functional.normalize(output, p=2, dim=1)
        logit= output @ self.classifier.proto
        return logit

class AttnMaskedFusion(nn.Module):
    """
    Attention Masked Fusion
    calculate attention weights for each modality and apply the mask
    """
    def __init__(self, classifier, input_dim=512):
        super(AttnMaskedFusion, self).__init__()
        self.attn_layer = nn.Sequential(
            nn.Linear(input_dim * 3, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 3),
            nn.Softmax(dim=1)
        )
        self.classifier = classifier

    def forward(self, x, x_flag,
                y, y_flag,
                z, z_flag):
        
        x = x * x_flag.unsqueeze(1) # flag 0 means None -> multiply 0
        y = y * y_flag.unsqueeze(1)
        z = z * z_flag.unsqueeze(1)
        
        combined = torch.cat([x, y, z], dim=1)  # B, D*3
        attn_weights = self.attn_layer(combined)  # B, 3
        
        attn_x = attn_weights[:, 0].unsqueeze(1) * x
        attn_y = attn_weights[:, 1].unsqueeze(1) * y
        attn_z = attn_weights[:, 2].unsqueeze(1) * z
        
        output = attn_x + attn_y + attn_z
        output = torch.nn.functional.normalize(output, p=2, dim=1)
        logit= output @ self.classifier.proto
        return logit