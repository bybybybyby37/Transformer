# official_translation.py (Final Colab Version)
import torch
import torch.nn as nn
import time
import math
import os
import json
import random
import warnings
import heapq
import sacrebleu
from types import SimpleNamespace
from tqdm import tqdm
from functools import partial
from torch.utils.tensorboard import SummaryWriter

# ignore console warnings
warnings.filterwarnings("ignore")

from models import model as my_model
from models.data_interface import create_iwslt17_dataloaders

# -----------------------------
# Auxiliary functions
# -----------------------------
def translate(model, src_sentence, sp_processor, device, max_len):
    model.eval()
    
    ids = sp_processor.EncodeAsIds(src_sentence)

    src_ids = [sp_processor.bos_id()] + ids + [sp_processor.eos_id()]
    
    src = torch.tensor(src_ids).unsqueeze(0).to(device)
    
    # Decoder input
    tgt_tokens = [sp_processor.bos_id()]
    
    for i in range(max_len):
        tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(src, tgt_tensor)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        
        if next_token == sp_processor.eos_id():
            break
        tgt_tokens.append(next_token)
        
    # SentencePiece decode
    return sp_processor.DecodeIds(tgt_tokens[1:]) 


def evaluate(model, loader, criterion, vocab_size, device):
    model.eval()
    total_loss = 0      # record Loss (Val Loss) with smoothing
    total_ppl_loss = 0  # record Loss (for PPL calculation) without smoothing
    total_count = 0

    # Criterion to calculate PPL (without Label Smoothing)
    ppl_criterion = nn.CrossEntropyLoss(ignore_index=criterion.ignore_index, label_smoothing=0.0)
    
    with torch.no_grad():
        for src, tgt_in, tgt_out in loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            
            with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                logits = model(src, tgt_in)
                
                # calculate regular loss (with label smoothing, used for reporting validation loss)
                loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
                
                # calculate loss for PPL (without label smoothing)
                loss_clean = ppl_criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
            
            total_loss += loss.item()
            total_ppl_loss += loss_clean.item() # accumulate clean loss (no smoothing) for perplexity
            total_count += 1
            
    avg_loss = total_loss / max(total_count, 1)
    
    # NOTE: use clean loss (without smoothing) to compute perplexity
    avg_ppl_loss = total_ppl_loss / max(total_count, 1)
    
    try:
        ppl = math.exp(avg_ppl_loss)
    except OverflowError:
        ppl = float('inf')
        
    return avg_loss, ppl


@torch.no_grad()
def beam_search_translate(model, src_sentence, sp_processor, device, max_len=128, beam_width=5, alpha=0.6):
    """
    Performs Beam Search translation.
    Optimized with vectorization and encoder memory reuse.
    """
    model.eval()
    
    # 1. Preprocessing: Encode source sentence
    tokens = sp_processor.EncodeAsIds(src_sentence)
    src_ids = [sp_processor.bos_id()] + tokens + [sp_processor.eos_id()]
    
    # Create tensor: [1, src_len]
    src_tensor = torch.tensor([src_ids], device=device) 
    
    # Generate source padding mask
    src_mask = model.make_src_mask(src_tensor) 
    
    # 2. Run Encoder (Only Once)
    # Get the encoder memory: [1, src_len, d_model]
    memory = model.encode(src_tensor, src_mask) 
    
    # 3. Prepare Beam Search (Vectorized)
    # Expand memory to match beam width: [beam_width, src_len, d_model]
    memory = memory.expand(beam_width, -1, -1)
    # Correct for nn.Transformer API
    src_mask = src_mask.expand(beam_width, -1)
    
    # Initialize sequences with BOS token: [beam_width, 1]
    cur_sequences = torch.full((beam_width, 1), sp_processor.bos_id(), device=device)
    
    # Initialize scores
    # The first beam has score 0, others -inf to force starting from the first beam
    beam_scores = torch.zeros(beam_width, device=device)
    beam_scores[1:] = -1e9 
    
    final_results = []
    
    # 4. Step-by-step Generation
    for step in range(max_len):
        # Create causal mask for the decoder
        tgt_mask = model.make_tgt_mask(cur_sequences)
        
        # Run Decoder (single pass for all beams)
        outputs = model.decode(cur_sequences, memory, src_mask, tgt_mask)
        
        # Get logits for the last token: [beam_width, vocab_size]
        logits = outputs[:, -1, :] 
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Calculate total scores: [beam_width, vocab_size]
        # Broadcasting: (beam_width, 1) + (beam_width, vocab_size)
        total_scores = beam_scores.unsqueeze(1) + log_probs
        
        # Select top-k candidates across all beams
        # Flatten view to find global top-k
        top_scores, top_indices = torch.topk(total_scores.view(-1), beam_width)
        
        # Decouple indices to find origin beam and new token
        prev_beam_indices = top_indices // model.tgt_vocab_size
        next_token_ids = top_indices % model.tgt_vocab_size
        
        # Update sequences: Append new tokens to selected beams
        cur_sequences = torch.cat([cur_sequences[prev_beam_indices], next_token_ids.unsqueeze(1)], dim=1)
        
        # Update scores
        beam_scores = top_scores
        
        # 5. Check for EOS (End of Sentence)
        is_eos = (next_token_ids == sp_processor.eos_id())
        if is_eos.any():
            for i in range(beam_width):
                if is_eos[i]:
                    # Apply length penalty: score / (length ^ alpha)
                    s = beam_scores[i] / (cur_sequences[i].size(0) ** alpha)
                    final_results.append((s, cur_sequences[i].tolist()))
                    
                    # Mark this beam as finished by setting score to -inf
                    beam_scores[i] = -1e9 
                    
        # If all beams are finished, break early
        if (beam_scores < -1e8).all(): 
            break

    # 6. Finalize Results
    # If no sequence ended with EOS (reached max_len), take current candidates
    if not final_results:
        for i in range(beam_width):
            score = beam_scores[i] / (cur_sequences[i].size(0) ** alpha)
            final_results.append((score, cur_sequences[i].tolist()))
            
    # Select the sequence with the highest score
    best_seq = max(final_results, key=lambda x: x[0])[1]
    
    # Decode IDs to string (removing BOS token at index 0)
    return sp_processor.DecodeIds(best_seq[1:])

# -----------------------------
# Main Functions
# -----------------------------
def main():
    # Load Config
    CONFIG_PATH = os.path.join("config", "hyperparameters.json")
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Initializing Data Pipeline (SentencePiece)...")
    print(f"Device: {DEVICE} | Batch: {cfg.batch_size} | D_Model: {cfg.d_model}")

    sp, train_loader, val_loader, test_loader = create_iwslt17_dataloaders(
        vocab_size=getattr(cfg, 'vocab_size', 8000),
        max_src_len=cfg.max_len,
        max_tgt_len=cfg.max_len,
        batch_size=cfg.batch_size,
        num_workers=2,
        seed=getattr(cfg, 'seed', 1337)
    )
    
    PAD_IDX = sp.pad_id()
    VOCAB_SIZE = sp.GetPieceSize()
    
    # beam search parameters
    BEAM_WIDTH = getattr(cfg, 'beam_width', 5)
    BEAM_ALPHA = getattr(cfg, 'beam_alpha', 0.6)
    
    print(f"Vocab Size: {VOCAB_SIZE}, Pad Id: {PAD_IDX} | Beam: {BEAM_WIDTH}, Alpha: {BEAM_ALPHA}")
    
    # ensure the save path for model-Checkpoint
    drive_root = os.path.dirname(cfg.save_path) 
    log_dir = os.path.join(drive_root, "runs", f"exp_{time.strftime('%Y%m%d-%H%M')}")
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to Drive: {log_dir}")
    
    # Model
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

    # Optimizer & Scheduler
    # set lr to 1.0 so that Noam Scheduler will ACTUALLY handle it
    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9, weight_decay=cfg.weight_decay
    )
    
    def rate(step, model_size, factor, warmup):
        if step == 0: step = 1
        return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))

    scheduler_factor = getattr(cfg, 'scheduler_factor', 0.5)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: rate(step, cfg.d_model, factor=scheduler_factor, warmup=cfg.warmup)
    )
    
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX, label_smoothing=getattr(cfg, 'label_smoothing', 0.1)
    )

    # load Checkpoint and Resuming training
    best_val_loss = float('inf')
    global_step = 0
    total_start = time.time()
    start_epoch = 1

    if os.path.exists(cfg.save_path):
        print(f"Found checkpoint at {cfg.save_path}, resuming training...")
        checkpoint = torch.load(cfg.save_path, map_location=DEVICE)
        
        # Compatibility check
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

    # Training Loop
    for epoch in range(start_epoch, cfg.max_epochs + 1):
        transformer.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.max_epochs}", dynamic_ncols=True)

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
        
        # Validation
        val_loss, val_ppl = evaluate(transformer, val_loader, criterion, VOCAB_SIZE, DEVICE)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/ppl", val_ppl, epoch)
        
        # print Validation PPL
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

        # try translate "Hello world"
        print("  Sample:", translate(transformer, "Hello world.", sp, DEVICE, cfg.max_len))
        print("-" * 50)
    
    # ----------------------------------------------------
    # FINAL TEST REPORT
    # ----------------------------------------------------
    print("\n" + "="*20 + " FINAL TEST REPORT " + "="*20)
    
    # Quantitative Evaluate
    test_loss, test_ppl = evaluate(transformer, test_loader, criterion, VOCAB_SIZE, DEVICE)
    print(f"Test Set Results: Loss = {test_loss:.4f} | PPL = {test_ppl:.2f}")
    
    # Collect all predictions and reference translations
    all_preds = []
    all_refs = []

    print("Calculating BLEU score...")
    num_samples = 1000 
    subset_indices = range(min(num_samples, len(test_loader.dataset)))

    # iterate over the test set for inference
    for i in tqdm(subset_indices, desc="Translating", unit="sent"):
        # get raw text (bypass tensors)
        raw_item = test_loader.dataset.data[i]['translation']
        src_text = raw_item['en']
        tgt_text = raw_item['zh'] # reference translation
        
        # generate predictions using beam search
        with torch.no_grad():
             pred_text = beam_search_translate(transformer, src_text, sp, DEVICE, cfg.max_len, beam_width=BEAM_WIDTH, alpha=BEAM_ALPHA)
        
        all_preds.append(pred_text)
        all_refs.append(tgt_text)

    # Compute BLEU
    bleu = sacrebleu.corpus_bleu(all_preds, [all_refs], tokenize='zh')

    print(f"\n=========================================")
    print(f"TEST BLEU: {bleu.score:.2f}")
    print(f"Signature: {bleu}")
    print(f"=========================================")

    # Qualitative Evaluate (Random Samples)
    print("\nRandom Test Samples:")
    
    dataset_handle = test_loader.dataset 
    total_samples = len(dataset_handle)
    
    indices = random.sample(range(total_samples), k=min(5, total_samples))
    
    for idx in indices:
        raw_item = dataset_handle.data[idx]['translation']
        src_raw = raw_item['en']
        tgt_raw = raw_item['zh']
        
        pred = beam_search_translate(transformer, src_raw, sp, DEVICE, cfg.max_len, beam_width=BEAM_WIDTH, alpha=BEAM_ALPHA)
        
        print(f"\n[Case {idx}]")
        print(f"  Src : {src_raw}")
        print(f"  Ref : {tgt_raw}")
        print(f"  Pred: {pred}")
    print("="*60)
    
    writer.close()

if __name__ == "__main__":
    main()