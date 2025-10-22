import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

from src.core.Linear_backbone import LinearBackbone

from tqdm import tqdm


class Linear_Trainer(nn.Module) : 
    # SimCLR
    def __init__(self, config, tokenizer, logger) : 
        super(Linear_Trainer, self).__init__()
        self.linear_backbone = LinearBackbone(config, tokenizer)
        self.tokenizer = tokenizer
        self.logger = logger
        
        if config.etf_path.exists() :         
            proto = torch.load(config.etf_path)
        else : 
            print("ETF not found, creating new one.")
            proto = self.linear_backbone.classifier.proto
            torch.save(proto, config.etf_path)
            
        self.linear_backbone.classifier.load_proto(proto)
        
        self.temperature = float(config.linear_temperature)
        self.device = "cuda"
        
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr = config.linear_lr,
            weight_decay = config.linear_weight_decay
        )
        self.criterion_no_reduction = nn.CrossEntropyLoss(reduction='none')
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, epoch, train_loader) : 
        self.train()
        self.linear_backbone.train()

        total_loss = 0.0
        n_batches = 0
        
        for batch in tqdm(train_loader, desc="Training", ncols=100, total=len(train_loader), leave=False) :
            features = self.linear_backbone(**batch)
            labels = batch['label'].to(self.device, non_blocking=True)
            
            features = features / features.norm(dim=1, keepdim=True)
            probs = features @ self.linear_backbone.classifier.proto
            probs = probs / self.temperature
            # check labels dist
            
            loss = self.criterion(probs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.mean().item()
            n_batches += 1
            
            metrics = {
                'train/loss' : float(loss.mean().item())
            }
            self.logger.log_metrics(metrics, step=epoch*len(train_loader)+n_batches, epoch=epoch)
        # self.scheduler.step()
        return total_loss / n_batches

    @torch.no_grad()
    def evaluate(self, val_loader) : 
        self.eval()
        self.linear_backbone.eval()
        
        total_loss = 0.0
        n_batches = 0
        output_collection = {}
        total_count, total_correct = 0, 0
        total_probs, total_labels = [], []
        
        for batch in val_loader : 
            features = self.linear_backbone(**batch)
            labels = batch['label'].to(self.device, non_blocking=True)
            
            features = features / features.norm(dim=1, keepdim=True)
            probs = features @ self.linear_backbone.classifier.proto
            probs = probs / self.temperature            
            adm_ids = batch['id']
            
            loss = self.criterion_no_reduction(probs, labels)
            
            labels = labels.detach().cpu().numpy()
            probs = probs.detach().cpu().numpy()
            
            for i, adm_id in enumerate(adm_ids) : 
                output_collection[adm_id] = {
                    'labels': labels[i],
                    'features': features[i].detach().cpu().numpy(),
                    'probs': probs[i],
                    'loss': loss[i].item()
                }

            total_loss += loss.mean().item()
            n_batches += 1
            total_count += labels.shape[0]
            total_correct += (probs.argmax(axis=1) == labels).sum().item()
            
            total_probs.append(probs)
            total_labels.append(labels)
            
            # INPUT CONTAINS NONE ERROR FIX
        total_probs = np.concatenate(total_probs, axis=0)
        total_labels = np.concatenate(total_labels, axis=0)
        # check nan in the probs and its corresponding labels, indexes
        if np.isnan(total_probs).any() :
            print("NaN values found in total_probs. Removing NaN entries.")
            valid_indices = ~np.isnan(total_probs).any(axis=1)
            total_probs = total_probs[valid_indices]
            total_labels = total_labels[valid_indices]
        auroc = roc_auc_score(total_labels, total_probs[:,1])
        
        auprc = average_precision_score(total_labels, total_probs[:,1])
        
        accuracy = total_correct / total_count
        
        output = {
            'loss' : total_loss / n_batches,
            'accuracy' : accuracy,
            'auroc' : auroc,
            'auprc' : auprc,
            'outputs' : output_collection
        }
        return output