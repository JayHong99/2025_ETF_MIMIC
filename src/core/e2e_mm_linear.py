import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
import numpy as np

from src.core.E2E_MM_backbone import MMBackbone

from tqdm import tqdm


class Linear_Trainer(nn.Module) : 
    def __init__(self, config, tokenizer, logger) : 
        super(Linear_Trainer, self).__init__()
        self.linear_backbone = MMBackbone(config, tokenizer)
        
        self.temperature = float(config.linear_temperature)
        self.device = "cuda"
        self.linear_backbone.to(self.device)
        self.linear_backbone.device = self.device
        
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr = config.linear_lr,
            weight_decay = config.linear_weight_decay
        )
        self.criterion_no_reduction = nn.CrossEntropyLoss(reduction='none')
        self.criterion = nn.CrossEntropyLoss()
        
        self.logger = logger # commet logger
    
    def train_epoch(self, epoch, train_loader) : 
        self.train()
        self.linear_backbone.train()

        total_loss = 0.0
        n_batches = 0
        
        for batch in tqdm(train_loader, desc="Training", ncols=100, total=len(train_loader), leave=False) :
            logits = self.linear_backbone(**batch)
            probs = logits / self.temperature
            labels = batch['label'].to(self.device, non_blocking=True).squeeze()
                        
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
            adm_ids = batch['id']            
            
            logits = self.linear_backbone(**batch)
            probs = logits / self.temperature
            probs = torch.softmax(probs, dim=1)
            labels = batch['label'].to(self.device, non_blocking=True).squeeze()
            loss = self.criterion_no_reduction(probs, labels)
            
            probs = probs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            
            for i, adm_id in enumerate(adm_ids) : 
                output_collection[adm_id] = {
                    'labels': labels[i],
                    'probs': probs[i],
                    'loss': loss[i].item()
                }

            total_loss += loss.mean().item()
            n_batches += 1
            total_count += labels.shape[0]
            total_correct += (probs.argmax(axis=1) == labels).sum().item()
            
            total_probs.append(probs)
            total_labels.append(labels)
            
        total_probs = np.concatenate(total_probs, axis=0)
        total_labels = np.concatenate(total_labels, axis=0)
        auroc = roc_auc_score(total_labels, total_probs[:,1])
        precision, recall, _ = precision_recall_curve(total_labels, total_probs[:,1])
        auprc = auc(recall, precision)

        accuracy = total_correct / total_count
        
        output = {
            'loss' : total_loss / n_batches,
            'accuracy' : accuracy,
            'auroc' : auroc,
            'auprc' : auprc,
            'outputs' : output_collection
        }
        return output