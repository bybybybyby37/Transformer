import torch
import torch.nn as nn
import time
import math
import os
import json
import random
import warnings
import sacrebleu
import numpy as np
from types import SimpleNamespace
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datasets import load_dataset


warnings.filterwarnings("ignore")


from models import model as my_model

from models.data_interface import (
    IWSLT17EnZhDataset, 
    load_or_train_spm_for_iwslt17, 
    collate_translation_batch
)
# use function in official_translation.py
from official_translation import evaluate, translate, beam_search_translate


def load_synthetic_data_as_list(file_path):
    print(f"[Data] Loading synthetic data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()] 
    
    en_lines = lines[0::2]
    zh_lines = lines[1::2]
    
    assert len(en_lines) == len(zh_lines), f"Error: Synthetic data mismatch! En: {len(en_lines)}, Zh: {len(zh_lines)}"
    
    data_list = []
    for en, zh in zip(en_lines, zh_lines):
        data_list.append({
            'translation': {
                'en': en,
                'zh': zh
            }
        })
        
    print(f"[Data] Loaded {len(data_list)} unique synthetic pairs.")
    return data_list


def mix_data_with_ratio(original_list, synth_list, ratio=10, seed=1337):
    """
    synth and mix training.
    ratio=10 means -> Real : Synth = 10 : 1
    """
    n_real = len(original_list)
    n_synth = len(synth_list)
    
    target_n_synth = int(n_real / ratio)
    
    print(f"[Mixing] Target Ratio {ratio}:1")
    print(f"  - Real Data: {n_real}")
    print(f"  - Original Synth: {n_synth}")
    print(f"  - Target Synth (Upsampled): {target_n_synth}")
    
    if n_synth < target_n_synth:
        repeat_factor = math.ceil(target_n_synth / n_synth)
        upsampled_synth = (synth_list * repeat_factor)[:target_n_synth]
        print(f"  - Action: Upsampled synthetic data {repeat_factor}x times.")
    else:
        upsampled_synth = synth_list[:target_n_synth]
        print(f"  - Action: Downsampled synthetic data.")
        
    mixed_list = original_list + upsampled_synth
    
    # Global Shuffle
    random.seed(seed)
    random.shuffle(mixed_list)
    
    print(f"[Mixing] Final Mixed Dataset Size: {len(mixed_list)} (Real + Synth)")
    return mixed_list

# -----------------------------
# Main Function
# -----------------------------
def main():
    CONFIG_PATH = os.path.join("config", "hyperparameters.json")
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    FT_EPOCHS = 15
    SYNTH_DATA_PATH = "data/synth_kept.en_zh.txt"
    CHECKPOINT_PATH = "checkpoints/transformer_en_zh.pt"
    SAVE_PATH = "checkpoints/best_model_synth.pt"
    MIX_RATIO = 10  # 10:1
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Initializing Data Pipeline for Mixed Training...")
    

    sp = load_or_train_spm_for_iwslt17(vocab_size=getattr(cfg, 'vocab_size', 8000))
    PAD_IDX = sp.pad_id()
    VOCAB_SIZE = sp.GetPieceSize()

    # IWSLT
    print("[Data] Loading original IWSLT2017 dataset...")
    dataset_dict = load_dataset("IWSLT/iwslt2017", "iwslt2017-en-zh", trust_remote_code=True)
    original_train_list = list(dataset_dict["train"])
    
    # data from LLM
    if os.path.exists(SYNTH_DATA_PATH):
        synth_data_list = load_synthetic_data_as_list(SYNTH_DATA_PATH)
    else:
        raise FileNotFoundError(f"Synthetic data not found at {SYNTH_DATA_PATH}")

    mixed_train_list = mix_data_with_ratio(
        original_train_list, 
        synth_data_list, 
        ratio=MIX_RATIO, 
        seed=getattr(cfg, 'seed', 1337)
    )
    
    # Dataset and DataLoader
    # train_loader with shuffle=True, Batch level random secured
    train_set = IWSLT17EnZhDataset(mixed_train_list, sp, cfg.max_len, cfg.max_len, "en", "zh")
    val_set = IWSLT17EnZhDataset(dataset_dict["validation"], sp, cfg.max_len, cfg.max_len, "en", "zh")
    test_set = IWSLT17EnZhDataset(dataset_dict["test"], sp, cfg.max_len, cfg.max_len, "en", "zh")
    
    collate_fn = lambda batch: collate_translation_batch(batch, pad_id=PAD_IDX)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    print("Initializing Transformer...")
    transformer = my_model.TransformerSeq2Seq(
        src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE,
        d_model=cfg.d_model, n_heads=cfg.n_heads,
        num_encoder_layers=cfg.num_encoder_layers, num_decoder_layers=cfg.num_decoder_layers,
        dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
        pad_idx=PAD_IDX
    ).to(DEVICE)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"[Train] Loading previous best checkpoint from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            transformer.load_state_dict(checkpoint['model_state_dict'])
            print(f"[Train] Weights loaded. Previous Best Val Loss: {checkpoint.get('best_val_loss', 'N/A')}")
        else:
            transformer.load_state_dict(checkpoint)
    else:
        print(f"[Warning] Checkpoint {CHECKPOINT_PATH} not found! Training from scratch.")

    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9, weight_decay=cfg.weight_decay
    )
    
    def rate(step, model_size, factor, warmup):
        if step == 0: step = 1
        return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))

    scheduler_factor = getattr(cfg, 'scheduler_factor', 0.5) 
    warmup_steps = 2000 # quick warmup with...
    fine_tune_factor = 0.1  # ... small learning rate
    fine_tune_warmup = 2000 
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: rate(step, cfg.d_model, factor=fine_tune_factor, warmup=fine_tune_warmup)
    )
    
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX, label_smoothing=getattr(cfg, 'label_smoothing', 0.1)
    )

    drive_root = os.path.dirname(cfg.save_path) 
    log_dir = os.path.join(drive_root, "runs", f"synth_10to1_{time.strftime('%Y%m%d-%H%M')}")
    writer = SummaryWriter(log_dir=log_dir)

    print(f"Starting Fine-tuning for {FT_EPOCHS} epochs with 10:1 data mixing...")
    best_val_loss = float('inf') 
    global_step = 0

    for epoch in range(1, FT_EPOCHS + 1):
        transformer.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"FT Epoch {epoch}/{FT_EPOCHS}", dynamic_ncols=True)

        for src, tgt_in, tgt_out in pbar:
            src, tgt_in, tgt_out = src.to(DEVICE), tgt_in.to(DEVICE), tgt_out.to(DEVICE)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(DEVICE.type=='cuda')):
                logits = transformer(src, tgt_in)
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
        val_loss, val_ppl = evaluate(transformer, val_loader, criterion, VOCAB_SIZE, DEVICE)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/ppl", val_ppl, epoch)
        
        print(f"\nEpoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, SAVE_PATH)
            print(f"  [Saved] Best Synth Model to {SAVE_PATH}")

        print("  Sample:", translate(transformer, "Hello world.", sp, DEVICE, cfg.max_len))
        print("-" * 50)
    
    print("\n" + "="*20 + " FINAL SYNTH TEST REPORT " + "="*20)
    print(f"Loading best fine-tuned model from {SAVE_PATH}...")
    checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
    transformer.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_ppl = evaluate(transformer, test_loader, criterion, VOCAB_SIZE, DEVICE)
    print(f"Test Set Results: Loss = {test_loss:.4f} | PPL = {test_ppl:.2f}")
    
    all_preds = []
    all_refs = []
    print("Calculating BLEU score on Test Set...")
    
    BEAM_WIDTH = getattr(cfg, 'beam_width', 5)
    BEAM_ALPHA = getattr(cfg, 'beam_alpha', 0.6)
    # modify here to choose quick sample test
    NUM_TEST_SAMPLES = 200  
    subset_indices = range(min(NUM_TEST_SAMPLES, len(test_loader.dataset)))
    
    for i in tqdm(subset_indices, desc="Translating", unit="sent"):
        raw_item = test_loader.dataset.data[i]['translation']
        src_text = raw_item['en']
        tgt_text = raw_item['zh']
        
        with torch.no_grad():
             pred_text = beam_search_translate(transformer, src_text, sp, DEVICE, cfg.max_len, beam_width=BEAM_WIDTH, alpha=BEAM_ALPHA)
        
        all_preds.append(pred_text)
        all_refs.append(tgt_text)

    bleu = sacrebleu.corpus_bleu(all_preds, [all_refs], tokenize='zh')

    print(f"\n=========================================")
    print(f"TEST BLEU: {bleu.score:.2f}")
    print(f"Signature: {bleu}")
    print(f"=========================================")
    
    writer.close()

if __name__ == "__main__":
    main()