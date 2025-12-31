# tune.py (Teammate Interface Version)
import optuna
import torch
import torch.nn as nn
import json
import os
import shutil
from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter

from models.data_interface import create_iwslt17_dataloaders
from models import model as my_model

def evaluate_loss(model, loader, criterion, vocab_size, device):
    model.eval()
    total_loss = 0
    total_count = 0
    with torch.no_grad():
        for src, tgt_in, tgt_out in loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            
            with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                logits = model(src, tgt_in)
                loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
            
            total_loss += loss.item()
            total_count += 1
    return total_loss / max(total_count, 1)

def objective(trial):
    # ================= define tune search space =================
    
    d_model = 256   # 256 is better than 384 / 512 after multiple attemps
    n_heads = 4              # 64 dim per head (Standard)
    num_encoder_layers = 6   # 6 is better after multiple attemps
    num_decoder_layers = 6   # 6 is better after multiple attemps
    dim_feedforward = 1024   # should be 4 times d_model
    
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    
    scheduler_factor = trial.suggest_float("scheduler_factor", 0.05, 0.5)

    warmup = trial.suggest_int("warmup", 2000, 5000)
    
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
        
    cfg_overrides = {
        "d_model": d_model,
        "n_heads": n_heads,
        "num_encoder_layers": num_encoder_layers,
        "num_decoder_layers": num_decoder_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "scheduler_factor": scheduler_factor,
        "warmup": warmup,
        "batch_size": batch_size,
        "learning_rate": 1.0 
    }

    with open("config/hyperparameters.json", "r") as f:
        base_cfg = json.load(f)
    base_cfg.update(cfg_overrides)
    cfg = SimpleNamespace(**base_cfg)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # init GradScaler to avoid NaN problem
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())


    sp, train_loader, val_loader, _ = create_iwslt17_dataloaders(
        vocab_size=cfg.vocab_size,  
        max_src_len=cfg.max_len,
        max_tgt_len=cfg.max_len,
        batch_size=cfg.batch_size,
        num_workers=2
    )
    
    VOCAB_SIZE = sp.GetPieceSize()
    PAD_IDX = sp.pad_id()

    model = my_model.TransformerSeq2Seq(
        src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE,
        d_model=cfg.d_model, n_heads=cfg.n_heads,
        num_encoder_layers=cfg.num_encoder_layers, num_decoder_layers=cfg.num_decoder_layers,
        dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
        pad_idx=PAD_IDX
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    
    def rate(step, model_size, factor, warmup):
        if step == 0: step = 1
        return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: rate(step, cfg.d_model, factor=cfg.scheduler_factor, warmup=cfg.warmup)
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    
    # Quick Training wiht Pruning
    N_EPOCHS_PRUNING = 3
    
    model.train()
    for epoch in range(1, N_EPOCHS_PRUNING + 1):
        total_loss = 0
        
        for src, tgt_in, tgt_out in train_loader:
            src, tgt_in, tgt_out = src.to(DEVICE), tgt_in.to(DEVICE), tgt_out.to(DEVICE)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(DEVICE.type=='cuda')):
                logits = model(src, tgt_in)
                loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))
            
            # Scale Loss
            scaler.scale(loss).backward()
            
            # Unscale Gradients for Gradient clipping
            scaler.unscale_(optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            
            # Step Optimizer
            scaler.step(optimizer)
            
            # Update Scaler
            scaler.update()
            
            scheduler.step()
            
            # NaN check
            if not torch.isnan(loss):
                total_loss += loss.item()

        val_loss = evaluate_loss(model, val_loader, criterion, VOCAB_SIZE, DEVICE)
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss

if __name__ == "__main__":
    print("Starting hyperparameter tuning with Teammate's Interface...")
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    
    print("\n" + "="*20 + " TUNING FINISHED " + "="*20)
    print("Best params found:", study.best_params)
    
    config_path = "config/hyperparameters.json"
    
    with open(config_path, "r") as f:
        base_config = json.load(f)
        
    shutil.copy(config_path, config_path + ".bak")
    
    base_config.update(study.best_params)

    if 'max_tokens' in base_config:
        del base_config['max_tokens']
        
    with open(config_path, "w") as f:
        json.dump(base_config, f, indent=4)
        
    print("Config updated! Starting official training...")
    
    import torch
    torch.cuda.empty_cache()
    import official_translation
    official_translation.main()