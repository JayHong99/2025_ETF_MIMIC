from omegaconf import OmegaConf
import argparse
import os
from copy import copy

from torch.utils.data import DataLoader
import torch


from src.utils.utils import setup_seed, save_checkpoint, save_output
from src.utils.config import load_cfg
from src.dataset.mimic4_dataset_MM import MIMIC4Dataset_EMB
from src.dataset.utils import MIMIC4Collator_EMB
from src.core.mm_linear import Linear_Trainer
from src.utils.utils import processed_data_path, read_txt, load_pickle

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_path', type=str, default='configs/MM_mortality_90days_BS_512_hidden_512.yaml')
parser.add_argument('--device', type=int, default=0, help='device id to use')
parser.add_argument('--do_train', type=bool, default=True, help='True for training')
parser.add_argument('--seed', type=int, default=2026, help='random seed')
parser.add_argument('--fusion_method', type=str, default=None, help='fusion method for multimodal model', 
                    choices=[None, 'Sum', 'WeightedFusion', 'AttnMaskedFusion', 'GraphFusion'])
args = parser.parse_args()

def main() : 
    config = load_cfg(args.cfg_path, args=args)
    print(config)
    setup_seed(config.seed)    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.device)
    
    if config.linear_output_dir.joinpath('best_epoch.pkl').exists() and config.do_train :
        print(f"Output already exists in {config.linear_output_dir}, skipping...")
        return
    
    train_set = MIMIC4Dataset_EMB(config, split='train')
    valid_set = MIMIC4Dataset_EMB(config, split='valid')
    test_set = MIMIC4Dataset_EMB(config, split='test')    
    mimic4_collate_fn = MIMIC4Collator_EMB()

    train_loader = DataLoader(
                    train_set,
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=4,
                    collate_fn=mimic4_collate_fn,
                    pin_memory=True,
                    drop_last=False,
                    persistent_workers=True,
                    # prefetch_factor=2
                )
    valid_loader = DataLoader(
                    valid_set,
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=4,
                    collate_fn=mimic4_collate_fn,
                    pin_memory=True,
                    drop_last=False,
                    persistent_workers=True,
                    # prefetch_factor=2
                )
    
    test_loader = DataLoader(
                    test_set,
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=4,
                    collate_fn=mimic4_collate_fn,
                    pin_memory=True,
                    drop_last=False,
                    # prefetch_factor=2
                )
    
    # memory efficient
    
    # tokenizer = train_set.tokenizer
    model = Linear_Trainer(config).to('cuda')
    del mimic4_collate_fn, valid_set, train_set, test_set
    
        
    
    if config.do_train : # do_train is False == train    
        if (config.linear_output_dir / 'best_epoch.pkl').exists() :
            print(f"Output already exists in {config.linear_output_dir}, skipping...")
            return
        
        # Check resume
        model_dirs = list(config.linear_model_dir.glob('linear_epoch*.pth'))
        if len(model_dirs) > 0 :
            model_dirs = sorted(model_dirs, key=lambda x: int(x.stem.split('_')[1].replace('epoch', '')))
            ckpt = torch.load(model_dirs[-1], map_location='cpu')
            model.load_state_dict(ckpt)
            resume_epoch = int(model_dirs[-1].stem.split('_')[1].replace('epoch', ''))
            print(f"Resume from epoch {resume_epoch} with : {model_dirs[-1]}")
            del ckpt, model_dirs
        else :
            resume_epoch = 0
            print("Training from scratch...")
        
        best_valid_auroc = 0.0
        for epoch in range(resume_epoch, config.linear_epochs) :
            epoch_outputs = {}
            score_outputs = {}
            train_loss = model.train_epoch(train_loader)
            valid_output = model.evaluate(valid_loader)
            
            if valid_output['auroc'] > best_valid_auroc :
                best_valid_auroc = valid_output['auroc']
                model_save_name = f"linear_epoch{epoch+1:03d}_train{train_loss:.4f}_valid{valid_output['loss']:.4f}"
                save_checkpoint(config.linear_model_dir, model, model_save_name)
                for mode, loader in zip(['train', 'valid', 'test'], [train_loader, valid_loader, test_loader]) : 
                    if mode == 'valid' : 
                        output = valid_output
                    else : 
                        output = model.evaluate(loader)
                    epoch_outputs[mode] = output
                    print(f"[Modality : {config.modality}] [Epoch {epoch} / {config.linear_epochs}] {mode} AUROC : {output['auroc']:.4f} | {mode} AUPRC : {output['auprc']:.4f}")
                    
                    score_outputs[mode] = {
                        'accuracy' : output['accuracy'],
                        'auroc' : output['auroc'],
                        'auprc' : output['auprc'],
                        'loss' : output['loss']
                    }

                output_name = f"epoch{epoch+1:03d}"
                save_output(config.linear_output_dir, epoch_outputs, output_name)        
                save_output(config.linear_output_dir, score_outputs, output_name+'_scores')
                print(f"[Epoch {epoch} / {config.linear_epochs}] Saved outputs to {config.linear_output_dir / (output_name+'.pkl')}")
                print('='*100)
                print()
                # Memory cleanup
                del epoch_outputs, score_outputs, output, valid_output, train_loss
            else :
                print(f"[Epoch {epoch} / {config.linear_epochs}] Valid AUROC did not improve from {best_valid_auroc:.4f}. Skipping checkpoint save.")

if __name__ == '__main__':
    main()