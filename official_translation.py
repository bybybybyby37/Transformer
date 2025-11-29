# official_translation.py (Final Colab Version)
import torch
import torch.nn as nn
import time
import math
import os
import json
import random
import warnings
from types import SimpleNamespace
from tqdm import tqdm
from torch.utils.data import DataLoader
from functools import partial
from torch.utils.tensorboard import SummaryWriter

# 屏蔽烦人的警告
warnings.filterwarnings("ignore")

from models import tokenizer as my_tokenizer
from models import dataset as my_dataset
from models import model as my_model

# -----------------------------
# 辅助函数
# -----------------------------
def evaluate(model, loader, criterion, vocab_size, device):
    """
    计算验证集/测试集的 Loss 和 PPL
    """
    model.eval()
    total_loss = 0
    total_count = 0
    
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            
            # 混合精度推理
            with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                logits = model(src, tgt_input)
                loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
            
            total_loss += loss.item()
            total_count += 1
            
    avg_loss = total_loss / max(total_count, 1)
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float('inf')
        
    return avg_loss, ppl

def translate(model, src_sentence, tokenizer, device, max_len):
    """
    推理函数：输入英文，输出中文
    """
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

# -----------------------------
# 主程序
# -----------------------------
def main():
    # 1. Config 加载
    CONFIG_PATH = os.path.join("config", "hyperparameters.json")
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE} | Batch: {cfg.batch_size} | D_Model: {cfg.d_model}")
    
    # [修改点 3] TensorBoard 保存到 Google Drive
    # 假设 cfg.save_path 已经在 Drive 里了，我们把 log 也放进去
    drive_root = os.path.dirname(cfg.save_path) # e.g. /content/drive/MyDrive/Transformer_Checkpoints
    log_dir = os.path.join(drive_root, "runs", f"exp_{time.strftime('%Y%m%d-%H%M')}")
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to Drive: {log_dir}")

    # 2. Tokenizer & Data
    print(f"Loading Tokenizer...")
    tok = my_tokenizer.BPETokenizer(cfg.tokenizer_path)
    VOCAB_SIZE = tok.vocab_size
    PAD_IDX = tok.pad_token_id
    
    print("Loading Datasets...")
    train_ds = my_dataset.TranslationDataset(cfg.data_path_train, tok, max_len=cfg.max_len)
    
    # Validation Set
    if os.path.exists(cfg.data_path_valid):
        valid_ds = my_dataset.TranslationDataset(cfg.data_path_valid, tok, max_len=cfg.max_len)
    else:
        valid_ds = train_ds

    # [修改点 1] Test Set 加载 (用于最终测试)
    test_ds = None
    TEST_CSV = "data/test.csv" # 假设你的 test.csv 在这里
    if os.path.exists(TEST_CSV):
        print(f"Loading Test Set from {TEST_CSV}...")
        test_ds = my_dataset.TranslationDataset(TEST_CSV, tok, max_len=cfg.max_len)
    
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

    # 4. Optimizer & Scheduler
    # 注意：这里 lr 设为 1.0 交给 Noam Scheduler
    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9, weight_decay=cfg.weight_decay
    )
    
    def rate(step, model_size, factor, warmup):
        if step == 0: step = 1
        return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: rate(step, cfg.d_model, factor=1.0, warmup=cfg.warmup)
    )
    
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX, label_smoothing=getattr(cfg, 'label_smoothing', 0.0)
    )

    # 5. 断点续训与状态恢复
    best_val_loss = float('inf')
    global_step = 0
    total_start = time.time()
    start_epoch = 1

    if os.path.exists(cfg.save_path):
        print(f"Found checkpoint at {cfg.save_path}, resuming training...")
        checkpoint = torch.load(cfg.save_path, map_location=DEVICE)
        
        # 兼容性检查
        if 'model_state_dict' in checkpoint:
            transformer.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Resuming from Epoch {start_epoch}, Best Val Loss: {best_val_loss:.4f}")
        else:
            print("Legacy checkpoint detected (weights only). Starting fresh with loaded weights.")
            transformer.load_state_dict(checkpoint)
    else:
        print("No checkpoint found, starting from scratch.")

    # 6. Training Loop
    for epoch in range(start_epoch, cfg.max_epochs + 1):
        transformer.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.max_epochs}", dynamic_ncols=True)

        for src, tgt in pbar:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            optimizer.zero_grad()
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
            pbar.set_postfix({"loss": f"{loss_item:.4f}", "lr": f"{current_lr:.6f}"})
            global_step += 1

        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation
        val_loss, val_ppl = evaluate(transformer, valid_loader, criterion, VOCAB_SIZE, DEVICE)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/ppl", val_ppl, epoch)
        
        # [修改点 2] 打印 Validation PPL
        print(f"\nEpoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save Checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, cfg.save_path)
            print(f"  [Saved] Best Val Loss to Drive")

        # 简单采样 (Hello world)
        print("  Sample:", translate(transformer, "Hello world.", tok, DEVICE, cfg.max_len))
        print("-" * 50)
    
    # 7. 最终测试阶段 (Training 完成后)
    # ----------------------------------------------------
    if test_ds is not None:
        print("\n" + "="*20 + " FINAL TEST REPORT " + "="*20)
        test_loader = DataLoader(
            test_ds, batch_size=cfg.batch_size, shuffle=False, 
            collate_fn=partial(my_dataset.collate_fn, pad_idx=PAD_IDX),
            num_workers=WORKERS
        )
        
        # A. 定量评估 (Loss & PPL)
        test_loss, test_ppl = evaluate(transformer, test_loader, criterion, VOCAB_SIZE, DEVICE)
        print(f"Test Set Results: Loss = {test_loss:.4f} | PPL = {test_ppl:.2f}")
        
        # B. 定性评估 (随机抽取样本对比)
        print("\nRandom Test Samples:")
        indices = random.sample(range(len(test_ds)), k=min(5, len(test_ds))) # 随机抽5个
        for idx in indices:
            src_raw = test_ds.src_lines[idx]
            tgt_raw = test_ds.tgt_lines[idx]
            pred = translate(transformer, src_raw, tok, DEVICE, cfg.max_len)
            
            print(f"\n[Case {idx}]")
            print(f"  Src : {src_raw}")
            print(f"  Ref : {tgt_raw}")
            print(f"  Pred: {pred}")
        print("="*60)
    
    writer.close()

if __name__ == "__main__":
    main()