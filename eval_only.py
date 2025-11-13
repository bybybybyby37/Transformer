import torch, json, math, os
from types import SimpleNamespace
from models import tokenizer
from official_baseline import OfficialishCharLM, CharDataset 

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("config\\hyperparameters.json", "r") as f:
    cfg = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

data_path = cfg.data_path
split_ratio = tuple(cfg.split_ratio)
block_size = cfg.block_size
stride_overlap_ratio = cfg.stride_overlap_ratio
batch_size = cfg.batch_size 


with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

tok = tokenizer.CharTokenizer(text)
ids = tok.encode(text)
data = torch.tensor(ids, dtype=torch.long)

n = len(data)
n_train = int(split_ratio[0] * n)
n_val   = int(split_ratio[1] * n)
n_test  = n - n_train - n_val

test_data = data[n_train + n_val:]

my_stride = int(block_size * stride_overlap_ratio)
test_dataset = CharDataset(test_data, block_size, stride=my_stride)
test_loader  = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=0, 
    persistent_workers=False 
)

# load checkpoint
ckpt_path = "output/best_official.pt"
checkpoint = torch.load(ckpt_path, map_location=device)

cfg_model = checkpoint["config"]
model = OfficialishCharLM(
    vocab_size=cfg_model["vocab_size"],
    d_model=cfg_model["d_model"],
    n_heads=cfg_model["n_heads"],
    n_layers=cfg_model["n_layers"],
    d_ff=cfg_model["d_ff"],
    dropout=cfg_model["dropout"],
    block_size=cfg_model["block_size"],
).to(device)

model.load_state_dict(checkpoint["model"])
model.eval()

# run test
@torch.inference_mode()
def evaluate_simple(loader):
    model.eval()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)

test_loss = evaluate_simple(test_loader)
print(f"Loaded best model with best_val={checkpoint['best_val']:.4f} at epoch={checkpoint['epoch']}")
print(f"Final Test Loss: {test_loss:.4f}")
print(f"PPL: {math.exp(test_loss):.2f}")
print(f"BPC: {test_loss / math.log(2):.4f}")
