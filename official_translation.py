# official_translation.py (修复版)
import torch
import torch.nn as nn
import time
import math
import os
import json
import warnings # <--- 新增
from types import SimpleNamespace
from tqdm import tqdm
from torch.utils.data import DataLoader
from functools import partial
from torch.utils.tensorboard import SummaryWriter

# --- 1. 屏蔽烦人的警告 ---
warnings.filterwarnings("ignore") 
# -----------------------

from models import tokenizer as my_tokenizer
from models import dataset as my_dataset
from models import model as my_model

# -----------------------------
# 辅助函数
# -----------------------------
def evaluate(model, loader, criterion, vocab_size, device):
    """
    返回: (avg_loss, perplexity)
    """
    model.eval()
    total_loss = 0
    total_count = 0
    
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            
            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
            
            # 记录 batch 大小以计算更精确的加权平均 (可选，这里简单求平均)
            total_loss += loss.item()
            total_count += 1
            
    avg_loss = total_loss / max(total_count, 1)
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float('inf')
        
    return avg_loss, ppl

def translate(model, src_sentence, tokenizer, device, max_len):
    model.eval()
    src_ids = tokenizer.encode(src_sentence, add_special_tokens=True)
    src = torch.tensor(src_ids).unsqueeze(0).to(device)
    tgt_tokens = [tokenizer.sos_token_id]
    
    for i in range(max_len):
        tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(src, tgt_tensor)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        
        if next_token == tokenizer.eos_token_id:
            break
        tgt_tokens.append(next_token)
        
    return tokenizer.decode(tgt_tokens, skip_special_tokens=True)
def main():
    # 1. Config
    CONFIG_PATH = os.path.join("config", "hyperparameters.json")
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE} | Batch: {cfg.batch_size} | D_Model: {cfg.d_model}")
    
    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
    run_name = f"runs/transformer_en_zh_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=run_name)

    # 2. Tokenizer & Data
    print(f"Loading Tokenizer...")
    tok = my_tokenizer.BPETokenizer(cfg.tokenizer_path)
    VOCAB_SIZE = tok.vocab_size
    PAD_IDX = tok.pad_token_id
    
    print("Loading Datasets...")
    train_ds = my_dataset.TranslationDataset(cfg.data_path_train, tok, max_len=cfg.max_len)
    
    # 简单的验证集处理
    valid_ds = train_ds 
    if os.path.exists(cfg.data_path_valid):
        valid_ds = my_dataset.TranslationDataset(cfg.data_path_valid, tok, max_len=cfg.max_len)

    WORKERS = 0 

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, 
        collate_fn=partial(my_dataset.collate_fn, pad_idx=PAD_IDX),
        num_workers=WORKERS, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg.batch_size, shuffle=False, 
        collate_fn=partial(my_dataset.collate_fn, pad_idx=PAD_IDX),
        num_workers=WORKERS
    )

    # 3. Model
    print("Initializing Transformer...")
    transformer = my_model.TransformerSeq2Seq(
        src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE,
        d_model=cfg.d_model, n_heads=cfg.n_heads,
        num_encoder_layers=cfg.num_encoder_layers, num_decoder_layers=cfg.num_decoder_layers,
        dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
        pad_idx=PAD_IDX
    ).to(DEVICE)

    for p in transformer.parameters():
        if p.dim() > 1: nn.init.xavier_uniform_(p)

    # 4. Optimizer & Scheduler (关键修复！)
    # ---------------------------------------------------------
    # 修复点：这里 lr 必须设为 1.0，因为 Noam Scheduler 会输出真实的极小值。
    # 如果这里设为 0.0001，结果就是 0.0001 * 0.0005 = 0 (无法训练)
    optimizer = torch.optim.Adam(
        transformer.parameters(), 
        lr=1.0,  # <--- 强制改为 1.0
        betas=(0.9, 0.98), eps=1e-9, weight_decay=cfg.weight_decay
    )
    
    def rate(step, model_size, factor, warmup):
        if step == 0: step = 1
        return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: rate(step, cfg.d_model, factor=1.0, warmup=cfg.warmup)
    )
    # ---------------------------------------------------------
    
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX, 
        label_smoothing=getattr(cfg, 'label_smoothing', 0.0)
    )

    best_val_loss = float('inf')
    global_step = 0
    total_start = time.time()

    for epoch in range(1, cfg.max_epochs + 1):
        transformer.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.max_epochs}", dynamic_ncols=True)

        for src, tgt in pbar:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            optimizer.zero_grad()
            
            # 修复点：使用新的 torch.amp 语法，消除 warning
            with torch.amp.autocast('cuda', enabled=(DEVICE.type=='cuda')):
                logits = transformer(src, tgt_input)
                loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), cfg.grad_clip)
            
            optimizer.step()
            lr_scheduler.step()

            loss_item = loss.item()
            epoch_loss += loss_item
            current_lr = optimizer.param_groups[0]['lr']
            
            writer.add_scalar("train/loss", loss_item, global_step)
            writer.add_scalar("train/lr", current_lr, global_step)
            
            # 显示当前的 loss 和 lr
            pbar.set_postfix({
                "loss": f"{loss_item:.4f}", 
                "lr": f"{current_lr:.6f}"
            })
            global_step += 1

        # (Log Summary 代码保持不变...)
        avg_train_loss = epoch_loss / len(train_loader)
        val_loss, val_ppl = evaluate(transformer, valid_loader, criterion, VOCAB_SIZE, DEVICE)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/ppl", val_ppl, epoch)
        
        print(f"\nEpoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {time.time()-total_start:.0f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(transformer.state_dict(), cfg.save_path)
            print(f"  [Saved] Best Valid Loss")

        print("  Sample:", translate(transformer, "Hello world.", tok, DEVICE, cfg.max_len))
        print("-" * 50)
        
    writer.close()

if __name__ == "__main__":
    main()