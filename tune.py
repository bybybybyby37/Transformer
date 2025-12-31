# tune.py (Teammate Interface Version)
import optuna
import torch
import torch.nn as nn
import json
import os
import shutil
from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter

# 引入队友的数据接口
from models.data_interface import create_iwslt17_dataloaders
from models import model as my_model

def evaluate_loss(model, loader, criterion, vocab_size, device):
    model.eval()
    total_loss = 0
    total_count = 0
    with torch.no_grad():
        # [适配] 解包三个变量
        for src, tgt_in, tgt_out in loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            
            with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                logits = model(src, tgt_in)
                loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
            
            total_loss += loss.item()
            total_count += 1
    return total_loss / max(total_count, 1)

def objective(trial):
    # ================= 1. 定义智能搜索空间 =================
    
    # A. 决定模型的基础宽度 (256 vs 512)
    d_model = 256
    n_heads = 4              # 64 dim per head (Standard)
    num_encoder_layers = 6   # 回归 6 层
    num_decoder_layers = 6   # 回归 6 层
    dim_feedforward = 1024   # 4倍 d_model
    
    # 【正则化】：这是 Tune 的重点
    # 之前 0.12 左右表现不错，我们在附近细搜
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    
    # 【优化器】：寻找最佳收敛速度
    # 之前 Factor 1.08 有点激进，改为 0.05 - 0.5
    scheduler_factor = trial.suggest_float("scheduler_factor", 0.05, 0.5)

    warmup = trial.suggest_int("warmup", 2000, 5000)
    
    # Transformer 喜欢大 Batch，搜一下最佳平衡点
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
        
    # C. 其他超参数
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
    # =======================================================
    
    # 2. 加载基础配置
    with open("config/hyperparameters.json", "r") as f:
        base_cfg = json.load(f)
    base_cfg.update(cfg_overrides)
    cfg = SimpleNamespace(**base_cfg)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化 GradScaler (这是修复 NaN 的关键)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # 3. 数据加载 (切记：确保 create_iwslt17_dataloaders 里的 vocab_size 改回了 8000)
    # 建议直接在 tune.py 里写死 8000，防止 json 没改对
    sp, train_loader, val_loader, _ = create_iwslt17_dataloaders(
        vocab_size=8000,  # <--- 强制回退到 8000
        max_src_len=cfg.max_len,
        max_tgt_len=cfg.max_len,
        batch_size=cfg.batch_size,
        num_workers=2
    )
    
    VOCAB_SIZE = sp.GetPieceSize()
    PAD_IDX = sp.pad_id()

    # 4. 模型初始化
    model = my_model.TransformerSeq2Seq(
        src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE,
        d_model=cfg.d_model, n_heads=cfg.n_heads,
        num_encoder_layers=cfg.num_encoder_layers, num_decoder_layers=cfg.num_decoder_layers,
        dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
        pad_idx=PAD_IDX
    ).to(DEVICE)
    
    # 5. 优化器 & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    
    def rate(step, model_size, factor, warmup):
        if step == 0: step = 1
        return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: rate(step, cfg.d_model, factor=cfg.scheduler_factor, warmup=cfg.warmup)
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    
    # 6. 快速训练 (剪枝模式)
    N_EPOCHS_PRUNING = 3
    
    model.train()
    for epoch in range(1, N_EPOCHS_PRUNING + 1):
        total_loss = 0
        
        # [适配] 三元组解包
        for src, tgt_in, tgt_out in train_loader:
            src, tgt_in, tgt_out = src.to(DEVICE), tgt_in.to(DEVICE), tgt_out.to(DEVICE)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(DEVICE.type=='cuda')):
                logits = model(src, tgt_in)
                loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))
            
            # 使用 Scaler 进行反向传播和步进
            # 1. Scale Loss
            scaler.scale(loss).backward()
            
            # 2. Unscale Gradients (为了进行梯度裁剪，必须先 Unscale)
            scaler.unscale_(optimizer)
            
            # 3. 梯度裁剪 (现在是安全的，因为已经 Unscale 了)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            
            # 4. Step Optimizer
            scaler.step(optimizer)
            
            # 5. Update Scaler
            scaler.update()
            
            # Scheduler 正常步进
            scheduler.step()
            
            # 检查是否有 NaN (如果有，scaler.step 会跳过，loss 可能是 nan)
            if not torch.isnan(loss):
                total_loss += loss.item()

        # 验证
        val_loss = evaluate_loss(model, val_loader, criterion, VOCAB_SIZE, DEVICE)
        
        # 报告给 Optuna 进行剪枝
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss

if __name__ == "__main__":
    print("Starting hyperparameter tuning with Teammate's Interface...")
    
    # 1. 运行搜索
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30) # 在云端可以设大一点，比如 50
    
    print("\n" + "="*20 + " TUNING FINISHED " + "="*20)
    print("Best params found:", study.best_params)
    
    # 2. 保存并更新配置
    config_path = "config/hyperparameters.json"
    
    with open(config_path, "r") as f:
        base_config = json.load(f)
        
    shutil.copy(config_path, config_path + ".bak")
    
    # 更新配置
    base_config.update(study.best_params)
    # [注意] 清理掉之前可能残留的 max_tokens，因为现在用 batch_size 了
    if 'max_tokens' in base_config:
        del base_config['max_tokens']
        
    with open(config_path, "w") as f:
        json.dump(base_config, f, indent=4)
        
    print("Config updated! Starting official training...")
    
    # 3. 自动启动正式训练
    import torch
    torch.cuda.empty_cache()
    import official_translation
    official_translation.main()