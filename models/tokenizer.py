import os
import torch

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None

class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
    def encode(self, s):
        return [self.stoi[c] for c in s]
    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids)
    

class BPETokenizer:
    """BPE tokenizer for translation tasks."""
    def __init__(self, json_path="config/tokenizer_en_zh.json"):
        if Tokenizer is None:
            raise ImportError("Please install tokenizers: `pip install tokenizers`")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Tokenizer file not found at {json_path}. Run build_vocab.py first.")
            
        self.tokenizer = Tokenizer.from_file(json_path)
        
        # cache common special token IDs (used in training loop)
        self.pad_token_id = self.tokenizer.token_to_id("<pad>")
        self.unk_token_id = self.tokenizer.token_to_id("<unk>")
        self.sos_token_id = self.tokenizer.token_to_id("<sos>")
        self.eos_token_id = self.tokenizer.token_to_id("<eos>")
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text, add_special_tokens=True):
        """
        Encode text.
        If add_special_tokens=True, <sos> and <eos> will be added.
        """
        encoded = self.tokenizer.encode(text)
        ids = encoded.ids
        if add_special_tokens:
            # translation tasks usually require explicit structure: <sos> ... <eos>
            ids = [self.sos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        """
        Decode a list of IDs back to text.
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
