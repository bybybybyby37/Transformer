# models/dataset.py
import torch
import csv
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=128):
        """
        Args:
            csv_path: path to train.csv, validation.csv, or test.csv
            tokenizer: instance of BPETokenizer
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.src_lines = []
        self.tgt_lines = []
        
        print(f"Loading data from {csv_path}...")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None) # skip header (en, zh)
            
            for row in reader:
                if len(row) >= 2:
                    self.src_lines.append(row[0]) # English
                    self.tgt_lines.append(row[1]) # Chinese
                    
        self.lengths = [min(len(self.tokenizer.encode(s, add_special_tokens=False)) + 2, max_len) for s in self.src_lines]
        print(f"Loaded {len(self.src_lines)} pairs from {csv_path}.")

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_text = self.src_lines[idx]
        tgt_text = self.tgt_lines[idx]
        
        # Encoding (we control adding <sos>/<eos> in the dataset for safety; tokenizer may already add them)
        # assume tokenizer.encode(add_special_tokens=True) adds <sos>/<eos>
        src_ids = self.tokenizer.encode(src_text, add_special_tokens=True)
        tgt_ids = self.tokenizer.encode(tgt_text, add_special_tokens=True)

        if len(src_ids) == 0: src_ids = [self.tokenizer.unk_token_id] # prevent empty input
        if len(tgt_ids) == 0: tgt_ids = [self.tokenizer.unk_token_id]
        
        # truncation
        if len(src_ids) > self.max_len:
            src_ids = src_ids[:self.max_len]
            src_ids[-1] = self.tokenizer.eos_token_id # ensure ends with eos
            
        if len(tgt_ids) > self.max_len:
            tgt_ids = tgt_ids[:self.max_len]
            tgt_ids[-1] = self.tokenizer.eos_token_id

        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate_fn(batch, pad_idx):
    """
    Dynamic padding — keep as is; it's efficient for GPUs
    """
    src_batch, tgt_batch = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded