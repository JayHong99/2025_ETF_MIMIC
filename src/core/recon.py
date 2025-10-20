import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

import torch.optim as optim
from src.core.Recon_backbone import MMBackbone, DynamicNet

from tqdm import tqdm
import random


class Recon_Trainer(nn.Module) : 
    def __init__(self, config, tokenizer) : 
        super(Recon_Trainer, self).__init__()
        self.config = config
        self.pretrained_backbone = MMBackbone(config, tokenizer)
        proto = torch.load(config.etf_path)
        self.pretrained_backbone.classifier.load_proto(proto)
        
        # self.tabular_model = self.pretrained_backbone.tabular_model
        # self.note_model = self.pretrained_backbone.note_model
        # self.lab_model = self.pretrained_backbone.lab_model
        # self.classifier = self.pretrained_backbone.classifier
        self.proto = proto.to("cuda")
        
        
        self.dynamic_net = DynamicNet(
            c0 = random.random(),
            lr = config.boost_rate, 
            config = config
            )
        self.dynamic_net.add(model = self.pretrained_backbone.tabular_model, model_name = "tabular")
        self.dynamic_net.add(model = self.pretrained_backbone.note_model, model_name = "note")
        self.dynamic_net.add(model = self.pretrained_backbone.lab_model, model_name = "lab")
        print(f"Use Pre-trained Unimodal Models for ReconBoost")

        self.device = "cuda"
        self.ce_criterion = nn.CrossEntropyLoss()
        self.mse_criterion = nn.MSELoss()
        
        self.stage = 0
        self.model_names = []
        self.best_auroc = 0.0
    
    def _schedule_model(self, stage) : 
        if stage % 3 == 0:
            return "tabular"
        elif stage % 3 == 1:
            return "note"
        else:
            return "lab"
        
    
    def train_stage(self, stage, train_loader) : 
        self.train()
        self.dynamic_net.to_train()

        model_name = self._schedule_model(stage)
        print(f"Stage {stage}, pick {model_name} modality")
        
        modality_lr = self.config.m_lr
        ensemble_lr = self.config.e_lr
        
        # tabular note lab
        if model_name == 'tabular' : 
            model = self.pretrained_backbone.tabular_model
            pre_model = self.pretrained_backbone.lab_model
        elif model_name == 'note' :
            model = self.pretrained_backbone.note_model
            pre_model = self.pretrained_backbone.tabular_model
        elif model_name == 'lab' :
            model = self.pretrained_backbone.lab_model
            pre_model = self.pretrained_backbone.note_model
        
        self.dynamic_net.to_train()
        ce_loss = []
        optimizer = optim.SGD(model.parameters(), modality_lr, momentum=0.9, weight_decay=5e-4)
        
        stage_loss, stage_ga_loss, ce_loss = [], [], []
        
        for _ in range(self.config.epochs_per_stage) :
            for batch in tqdm(train_loader, desc=f"Training {model_name}", ncols=100, total=len(train_loader), leave=False) :
                # features = model(**batch)
                # features = features / features.norm(dim=1, keepdim=True)
                # probs = features @ self.proto
                labels = batch['label'].to(self.device, non_blocking=True)
                bs = labels.shape[0]
                
                join_features = self.dynamic_net.forward(batch, mask_model = model_name)
                join_features = join_features / join_features.norm(dim=1, keepdim=True)
                out_join = join_features @ self.proto
                # out_join = torch.as_tensor(out_join, dtype=torch.float32).cuda()
                
                obj_features = model(**batch)
                obj_features = obj_features / obj_features.norm(dim=1, keepdim=True)
                out_obj = obj_features @ self.proto
                target = torch.zeros(bs,self.config.num_classes).cuda().scatter_(1,labels.view(-1,1),1)
            
                boosting_loss = - self.config.weight1 *  (target * out_obj.log_softmax(1)).mean(-1) \
                                + self.config.weight2 * (target*out_join.detach().softmax(1) * out_obj.log_softmax(1)).mean(-1)
                model.zero_grad()
                
                if stage == 0 : 
                    loss = boosting_loss
                else : 
                    pre_obj_features = pre_model(**batch)
                    pre_obj_features = pre_obj_features / pre_obj_features.norm(dim=1, keepdim=True)
                    pre_out_obj = pre_obj_features @ self.proto
                    ga_loss = self.mse_criterion(out_obj.detach().softmax(1), pre_out_obj.detach().softmax(1)) ## ga loss
                    stage_ga_loss.append(ga_loss.mean().item())
                    loss = boosting_loss + self.config.alpha * ga_loss
                loss.mean().backward()
                
                optimizer.step()
                stage_loss.append(boosting_loss.mean().item())
            
        stage_mean_loss = np.mean(stage_loss).item()
        stage_mean_ga_loss = np.mean(stage_ga_loss).item()

        models_name = self.dynamic_net.get_model_name()
        print(f"There are {len(models_name)} modality(ies),{models_name}",)
        print(f"Adding {model_name} modality.....")

        self.dynamic_net.add(model, model_name)

        if stage >= 0 :
            optimizer_correct = optim.SGD(self.dynamic_net.parameters(), ensemble_lr, momentum=0.9, weight_decay=5e-4)
            for epoch in range(self.config.correct_epoch):
                for i, batch in enumerate(train_loader):
                    features = self.dynamic_net.forward_grad(batch)
                    features = features / features.norm(dim=1, keepdim=True)
                    out = features @ self.proto
                    labels = batch['label'].to(self.device, non_blocking=True).view(-1).long()
                    loss = self.ce_criterion(out, labels)
                    optimizer_correct.zero_grad()
                    loss.backward()
                    ce_loss.append(loss.item())
                    optimizer_correct.step()
        
        ce_mean_loss = np.mean(ce_loss).item()
        print(f"Results from stage {stage} : stage_loss {stage_mean_loss:.4f}, ga_loss {stage_mean_ga_loss:.4f}, ce_loss {ce_mean_loss:.4f}\n")
        return stage_mean_loss, ce_mean_loss

    def valid_stage(self, valid_loader, check_best = True) : 
        self.eval()
        self.dynamic_net.to_eval()
        all_labels, all_probs, all_preds = [], [], []
        output_collection = {}
        for _, batch in enumerate(tqdm(valid_loader, desc="Evaluating", ncols=100, total=len(valid_loader), leave=False)) :
            with torch.no_grad() : 
                features = self.dynamic_net.forward(batch)
                features = features / features.norm(dim=1, keepdim=True)
                out = features @ self.proto
                labels = batch['label'].to(self.device, non_blocking=True).view(-1).long()
                probs = F.softmax(torch.as_tensor(out, dtype=torch.float32).cuda(), dim=1)
                _, preds = torch.max(probs, 1)
                
                
                all_labels.append(labels.cpu())
                all_probs.append(probs.cpu())
                all_preds.append(preds.cpu())
                
                adm_ids = batch['id']
                for i, adm_id in enumerate(adm_ids) :
                    output_collection[adm_id] = {
                        "labels" : labels[i].cpu().item(),
                        'features' : features[i].cpu().numpy(),
                        "probs" : probs[i].cpu().numpy(),
                    }
                
        auroc = roc_auc_score(torch.cat(all_labels).numpy(), torch.cat(all_probs).numpy()[:,1])
        auprc = average_precision_score(torch.cat(all_labels).numpy(), torch.cat(all_probs).numpy()[:,1])
        acc = (torch.cat(all_labels).numpy() == torch.cat(all_preds).numpy()).mean()
        if auroc > self.best_auroc and check_best:
            self.best_auroc = auroc
        output_collection['auroc'] = auroc
        output_collection['auprc'] = auprc
        output_collection['accuracy'] = acc
        return output_collection
