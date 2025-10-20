from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig
from pathlib import Path


@dataclass
class TrainConfig:
    seed: int = 2026
    modality: str = 'tabular'
    task: str = 'mortality'
    target_days: int = 7
    num_classes: int = 2
    cfg_path: str = 'configs/ssl_tabular.yaml'
    do_train: bool = True
    
    ## Model Training
    device: int = 0
    batch_size: int = 256
    num_workers: int = 4
    bert_type: str = "prajjwal1/bert-tiny" # "emilyalsentzer/Bio_ClinicalBERT"
    code_pretrained_embedding: bool = True
    code_layers: int = 2
    code_heads: int = 2
    dropout: float = 0.25
    embedding_size: int = 512
    projection_size: int = 128
    rnn_layers: int = 2
    rnn_type: str = 'RNN'
    rnn_bidirectional: bool = True
    
    linear_lr: float = 1e-3
    linear_weight_decay: float = 1e-5
    linear_epochs: int = 100
    linear_temperature: float = 0.1
    num_stages: int = 100
    epochs_per_stage: int = 4
    correct_epoch: int = 4
    
    weight1: float = 1.0
    weight2: float = 0.25
    
    boost_rate: float = 1.0
    m_lr: float = 0.01
    e_lr: float = 0.01
    alpha: float = 0.5
    
    
    # ---- derived from YAML ----
    result_dir: Path = ""
    modality_dir: Path = ""
    exp_modality_name: str = ""
    exp_name: str = ""
    
    linear_run_dir: Path = ""
    linear_model_dir: Path = ""
    linear_output_dir: Path = ""
    
    etf_dir: Path = ""
    etf_path: Path = ""
    
    pretrained_tabular_path: Path = ""
    pretrained_lab_path: Path = ""
    pretrained_note_path: Path = ""
    
    tabular_run_dir: Path = ""
    lab_run_dir: Path = ""
    note_run_dir: Path = ""
    

def load_cfg(yaml_path, args) -> DictConfig:
    base = OmegaConf.structured(TrainConfig)
    user = OmegaConf.load(yaml_path)
    cfg = OmegaConf.merge(base, user)
    
    if args is not None : 
        cli = OmegaConf.create(vars(args))
        cfg = OmegaConf.merge(cfg, cli)

    OmegaConf.set_readonly(cfg, True)
    
    for key, value in cfg.items() : 
        if str(key).endswith('_dir') and isinstance(value, Path) : 
            value.mkdir(parents=True, exist_ok=True)
    
    
    return cfg