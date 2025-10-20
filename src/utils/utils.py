from pathlib import Path
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
import pickle
import os

remote_root = Path("/home/data/2025_MIMICIV_processed")
processed_data_path = remote_root

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
    
def read_txt(filename):
    data = []
    with open(filename, "r") as file:
        lines = file.read().splitlines()
        for line in lines:
            data.append(line)
    return data


# def save_checkpoint(model_save_dir, model, name):
#     state_dict = model.state_dict()
#     path = model_save_dir / f"{name}.pth"
#     torch.save(state_dict, path)
#     return

def save_checkpoint(model_save_dir, model, name):
    with torch.no_grad():
        try : 
            sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        except : # already given as model.state_dict()
            sd = {k: v.detach().cpu() for k, v in model.items()}
    tmp = Path("tmp") / f".{name}.tmp"
    dst = model_save_dir / f"{name}.pth"
    with open(tmp, "wb", buffering=8*1024*1024) as f:
        torch.save(sd, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, dst)

# def save_checkpoint(model_save_dir, model, name):
#     # save only trainable parameters
#     ## VIT or BERT style model -> we use pretrained model and only train a few linear layers
#     state_dict = {k: v for k, v in model.state_dict().items() if v.requires_grad}
#     path = model_save_dir / f"{name}.pth"
#     torch.save(state_dict, path)
#     return

def save_output(output_dir, output, name):
    path = output_dir / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(output, f)
    return