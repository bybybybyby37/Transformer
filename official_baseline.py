# official_baseline.py (fixed)
# Decoder-only causal LM built with PyTorch's nn.TransformerEncoder.

import os, math, time, json, requests
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import torch.multiprocessing as mp
import gc

from models import tokenizer

# -----------------------------
# Load config
# -----------------------------
with open(os.path.join("config", "hyperparameters.json"), "r") as f:
    cfg = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

data_path = cfg.data_path
save_path = cfg.save_path.replace(".pt", "_official.pt")  # avoid overwriting your original
split_ratio = tuple(cfg.split_ratio)
block_size = cfg.block_size
batch_size = cfg.batch_size
patience = cfg.patience
max_epochs = cfg.max_epochs
# eval_interval_epochs = cfg.eval_interval  # Not used - validation happens every epoch
stride_overlap_ratio = cfg.stride_overlap_ratio

d_model = cfg.d_model
n_heads = cfg.n_heads
n_layers = cfg.n_layers
d_ff = cfg.d_ff
dropout = cfg.dropout
weight_decay=cfg.weight_decay
warmup = cfg.warmup
learning_rate_scale = cfg.learning_rate_scale

grad_clip = cfg.grad_clip
seed = cfg.seed

device = "cuda" if torch.cuda.is_available() else "cpu"

def data_fetch():
    # -----------------------------
    # Dataset fetch
    # -----------------------------
    os.makedirs("data", exist_ok=True)
    if data_path == "data/tiny_shakespeare.txt":
        if os.path.exists(data_path):
            print(f"'{data_path}' already exists, skipping download.")
        else:
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            text = requests.get(url).text
            with open("data/tiny_shakespeare.txt", "w", encoding="utf-8") as f:
                f.write(text)
            print("Tiny Shakespeare downloaded! File size:", len(text), "characters")
    elif data_path == "data/full_shakespeare.txt":
        if os.path.exists(data_path):
            print(f"'{data_path}' already exists, skipping download.")
        else:
            url = "https://www.gutenberg.org/files/100/100-0.txt"
            print("Downloading full Shakespeare from Project Gutenberg...")
            text = requests.get(url).text
            if "*** START" in text:
                text = text.split("*** START")[1]
            if "*** END" in text:
                text = text.split("*** END")[0]
            with open(data_path, "w", encoding="utf-8") as f:
                f.write(text)
            print("Full Shakespeare downloaded! File size:", len(text), "characters")
    else:
        raise SystemExit("Unexpected dataset, stop training.")


class CharDataset(torch.utils.data.Dataset):
    def __init__(self, data, block_size, stride=None):
        self.data = data
        self.block_size = block_size
        self.stride = self.block_size if (stride is None or stride < 1) else stride
        last_valid_start_idx = len(self.data) - self.block_size - 1
        self.num_samples = 0 if last_valid_start_idx < 0 else (last_valid_start_idx // self.stride) + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        actual_idx = idx * self.stride
        x = self.data[actual_idx : actual_idx + self.block_size]
        y = self.data[actual_idx + 1 : actual_idx + 1 + self.block_size]
        return x, y

# -----------------------------
# Official-ish Decoder-only Transformer using nn.TransformerEncoder (causal mask)
# -----------------------------
class OfficialishCharLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout, block_size):
        super().__init__()
        self.block_size = block_size
        self.d_model = d_model

        self.tok = nn.Embedding(vocab_size, d_model)
        # Use sinusoidal positional encoding (official Transformer architecture)
        self.register_buffer('pos_encoding', self._create_sinusoidal_encoding(block_size, d_model))

        # Use Pre-LN for better training stability (especially for deeper networks)
        # Pre-LN is more stable than Post-LN for 8+ layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True  # Pre-LN (more stable)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok.weight

    def _create_sinusoidal_encoding(self, max_len, d_model):
        """Create sinusoidal positional encoding as in original Transformer paper."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, d_model]

    def _causal_mask(self, T, device):
        # Causal mask: prevent attending to future positions
        # PyTorch TransformerEncoder src_mask: True positions are masked (set to -inf)
        # Upper triangular matrix with True on future positions
        mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, idx, targets=None, start_pos=0):
        """
        Args:
            idx: input token indices [B, T]
            targets: target token indices [B, T] (optional)
            start_pos: starting position for positional encoding (for generation)
        """
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.block_size}")

        # Token embeddings scaled by sqrt(d_model)
        x = self.tok(idx) * math.sqrt(self.d_model)
        
        # Add positional encoding with correct offset for generation
        pos_end = start_pos + T
        if pos_end <= self.block_size:
            x = x + self.pos_encoding[:, start_pos:pos_end]
        else:
            # Handle case where we exceed block_size (shouldn't happen in normal use)
            # Use modulo or truncate
            pos_indices = torch.arange(start_pos, pos_end, device=idx.device) % self.block_size
            x = x + self.pos_encoding[:, pos_indices]

        # Create causal mask
        causal = self._causal_mask(T, idx.device)  # [T, T]
        y = self.encoder(x, mask=causal)  # [B, T, C], encoder-only with causal mask
        y = self.ln_f(y)
        logits = self.lm_head(y)  # [B, T, V]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=1.0, top_k=None):
        self.eval()
        for i in range(max_new_tokens):
            # Get the context window (last block_size tokens)
            idx_cond = idx[:, -self.block_size:] if idx.size(1) > self.block_size else idx
            # Calculate start position for positional encoding
            start_pos = max(0, idx.size(1) - self.block_size)
            
            logits, _ = self(idx_cond, start_pos=start_pos)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, k=top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

# -----------------------------
# Noam scheduler wrapper
# -----------------------------
class NoamOpt:
    def __init__(self, d_model, warmup, optimizer, factor=1.0):
        self._step = 0
        self.warmup = warmup
        self.factor = factor * (d_model ** (-0.5))
        self.opt = optimizer
        self._lr = 0.0

    def step(self):
        self._step += 1
        # Standard Noam scheduler: linear warmup, then decay
        if self._step <= self.warmup:
            lr = self.factor * self._step / (self.warmup ** 1.5)
        else:
            lr = self.factor * (self._step ** (-0.5))
        for g in self.opt.param_groups:
            g['lr'] = lr
        self._lr = lr

    @property
    def lr(self):
        return self._lr

    def zero_grad(self):
        self.opt.zero_grad(set_to_none=True)
    
    def state_dict(self):
        """Return Noam internal state + underlying optimizer state."""
        return {
            "step": self._step,
            "lr": self._lr,
            "warmup": self.warmup,
            "factor": self.factor,
            "optimizer_state": self.opt.state_dict(),
        }

    def load_state_dict(self, state):
        """Restore saved Noam & optimizer state."""
        self._step = state["step"]
        self._lr = state["lr"]
        self.warmup = state["warmup"]
        self.factor = state["factor"]
        self.opt.load_state_dict(state["optimizer_state"])

        for g in self.opt.param_groups:
            g["lr"] = self._lr

def clean_gpu_memory(device):
    """Try to release cached GPU memory periodically."""
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()


def main():
    data_fetch()
    # -----------------------------
    # Tokenizer & splits
    # -----------------------------
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    tok = tokenizer.CharTokenizer(text)
    print(len(tok.chars), "unique chars")

    ids = tok.encode(text)
    data = torch.tensor(ids, dtype=torch.long)

    vocab_size = getattr(tok, "vocab_size", len(tok.chars))
    print("vocab_size =", vocab_size)
    mx = int(max(ids)) if len(ids) > 0 else -1
    assert mx < int(vocab_size), f"max id {mx} >= vocab_size {int(vocab_size)}"

    n = len(data)
    n_train = int(split_ratio[0] * n)
    n_val = int(split_ratio[1] * n)
    n_test = n - n_train - n_val

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    print(f"Total tokens: {n:,}")
    print(f"Train: {len(train_data):,}, Val: {len(val_data):,}, Test: {len(test_data):,}")

    my_stride = int(block_size * stride_overlap_ratio)
    train_dataset = CharDataset(train_data, block_size, stride=my_stride)
    val_dataset   = CharDataset(val_data,   block_size, stride=my_stride)
    test_dataset  = CharDataset(test_data,  block_size, stride=my_stride)
    num_workers = os.cpu_count() // 2 or 2
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  pin_memory=True, num_workers=num_workers, persistent_workers=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, persistent_workers=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, persistent_workers=True)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # -----------------------------
    # Model / Optimizer / Logger
    # -----------------------------
    torch.manual_seed(seed)
    model = OfficialishCharLM(vocab_size, d_model, n_heads, n_layers, d_ff, dropout, block_size).to(device)
    print("Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    run_dir = f"runs/official_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=run_dir)
    print("TensorBoard logdir:", run_dir)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # def noam_lr_lambda(step: int):
    #     if step <= 0:
    #         return 0.0
    #     return (d_model ** -0.5) * min(step ** -0.5, step * (warmup ** -1.5))

    adam = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9, weight_decay=weight_decay)
    # Reduce learning rate scale slightly for better stability with Pre-LN
    # Pre-LN can handle slightly higher LR, but we're being conservative
    base_opt = NoamOpt(d_model, warmup, adam, factor=learning_rate_scale)
    # Initialize learning rate (don't call step() here, it will be called after first backward)
    # Set initial learning rate to a small value to avoid issues
    initial_lr = base_opt.factor * 1.0 / (warmup ** 1.5) if warmup > 0 else base_opt.factor
    for g in adam.param_groups:
        g['lr'] = initial_lr
    base_opt._lr = initial_lr
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_val = float("inf")
    bad_epochs = 0
    global_step = 0

    @torch.inference_mode()
    def evaluate(loader):
        model.eval()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            # Use autocast for consistency with training
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                _, loss = model(xb, yb)
            total_loss += loss.item()
        clean_gpu_memory(device)
        model.train()
        return total_loss / max(len(loader), 1)


    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_start_time = time.time()

        # Gradient accumulation (set >1 if want to simulate larger batches)
        accum_steps = 1

        # tqdm progress bar for the current epoch
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{max_epochs}",
            dynamic_ncols=True,
            leave=False,  # keep only one dynamic line, no screen flooding
        )

        # Initialize gradients at the start of each epoch
        base_opt.zero_grad()

        for batch_idx, (xb, yb) in pbar:
            xb, yb = xb.to(device), yb.to(device)

            # Forward + loss
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                _, loss = model(xb, yb)
                loss = loss / accum_steps

            # Check for NaN/Inf loss before backward
            if torch.isnan(loss) or torch.isinf(loss):
                tqdm.write(f"Warning: NaN/Inf loss detected at batch {batch_idx}, skipping...")
                base_opt.zero_grad()
                continue

            # Backward (accumulate gradients)
            scaler.scale(loss).backward()

            # Only update optimizer and clip gradients at accumulation boundary
            if (batch_idx + 1) % accum_steps == 0:
                # Unscale gradients before clipping
                scaler.unscale_(adam)
                
                # Check for NaN/Inf gradients before clipping
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
                if math.isnan(grad_norm) or math.isinf(grad_norm):
                    tqdm.write(f"Warning: NaN/Inf gradient norm detected at batch {batch_idx}, skipping update...")
                    scaler.update()  # Update scaler even if we skip the step
                    base_opt.zero_grad()
                    continue
                
                # Optimizer step
                scaler.step(adam)
                scaler.update()
                # Update learning rate scheduler after optimizer step
                base_opt.step()
                base_opt.zero_grad()
                
                # Log learning rate only after it's been updated
                current_lr = base_opt.lr
                writer.add_scalar("lr", current_lr, global_step)
            else:
                # During accumulation, we can't compute accurate grad_norm without unscale
                # Use a placeholder value for logging
                grad_norm = 0.0
                # Use the last known learning rate for logging during accumulation
                current_lr = base_opt.lr

            # Detach scalar values for display/logging
            loss_item = loss.item() * accum_steps
            bpc = loss_item / math.log(2)
            ppl = math.exp(loss_item) if loss_item < 20 else float("inf")

            # Update tqdm line with key metrics
            pbar.set_postfix({
                "loss": f"{loss_item:.4f}",
                "bpc": f"{bpc:.3f}",
                "ppl": f"{ppl:.1f}" if ppl != float("inf") else "inf",
                "lr": f"{current_lr:.6f}",
                "gnorm": f"{grad_norm:.2f}",
            })

            # TensorBoard logging
            writer.add_scalar("loss/train_batch", loss_item, global_step)
            writer.add_scalar("grad_norm", grad_norm, global_step)
            global_step += 1

        # End of epoch summary
        tqdm.write(f"[Epoch {epoch}] time={time.time() - epoch_start_time:.1f}s")

        # ---- Validation phase ----
        val_loss = evaluate(val_loader)
        bpc = val_loss / math.log(2)
        tqdm.write(f"[Epoch {epoch}] Val Loss {val_loss:.4f} | Val BPC {bpc:.4f}")

        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("metrics/val_bpc", bpc, epoch)

        # ---- Early stopping and checkpointing ----
        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save({
                "model": model.state_dict(),
                "optimizer": base_opt.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "tok_chars": tok.chars,
                "config": dict(d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
                            dropout=dropout, block_size=block_size, vocab_size=vocab_size)
            }, save_path)
            tqdm.write(f"Improved! Best val_loss={best_val:.4f} (saved to {save_path})")
        else:
            bad_epochs += 1
            tqdm.write(f"No improvement ({bad_epochs}/{patience})")
            if bad_epochs >= patience:
                tqdm.write("Early stopping triggered.")
                break
        clean_gpu_memory(device)


    # -----------------------------
    # Test best checkpoint + a sample generation
    # -----------------------------
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    print(f"Loaded best model with best_val={checkpoint['best_val']:.4f} at epoch={checkpoint['epoch']}")

    @torch.inference_mode()
    def evaluate_simple(loader):
        model.eval()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            # Use autocast for consistency with training
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                _, loss = model(xb, yb)
            total_loss += loss.item()
        clean_gpu_memory(device)
        return total_loss / max(len(loader), 1)

    test_loss = evaluate_simple(test_loader)
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Perplexity (PPL): {torch.exp(torch.tensor(test_loss)):.2f}")
    print(f"BPC: {(test_loss / math.log(2)):.4f}")

    # Sample generation from a prompt
    prompt = "ROMEO:"
    start_ids = tok.encode(prompt)
    idx = torch.tensor([start_ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=400, temperature=1.0, top_k=None)
    print(tok.decode(out[0].tolist()))

if __name__ == "__main__":
    # Windows needs spawn + main-guard
    mp.freeze_support() 
    mp.set_start_method("spawn", force=True)
    main()