import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create a matrix of [max_len, d_model] representing the positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Compute the division term: 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as a buffer so it is part of the state_dict but not a trainable parameter
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        # Add positional encoding to the input embeddings
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads, 
                 num_encoder_layers, num_decoder_layers, dim_feedforward, 
                 dropout, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        # Store vocab size for external access (e.g., during beam search)
        self.tgt_vocab_size = tgt_vocab_size 

        # Embedding layers
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Official PyTorch Transformer module
        # Using batch_first=True to handle inputs of shape [batch, seq_len]
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        # Output projection layer
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        
        # Weight Tying: Share weights between target embedding and output projection
        # This is a standard practice to reduce parameters and improve performance
        self.generator.weight = self.tgt_tok_emb.weight

    # --- Helper methods for Mask creation ---
    
    def make_src_mask(self, src):
        """
        Creates a padding mask for the source sequence.
        Returns:
            Tensor: Boolean tensor where True indicates a padding token (to be ignored).
        """
        return (src == self.pad_idx)

    def make_tgt_mask(self, tgt):
        """
        Creates a causal mask (upper triangular matrix) for the target sequence
        to prevent positions from attending to subsequent positions.
        """
        return self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

    # --- Inference Methods (Decoupled for Efficiency) ---

    def encode(self, src, src_mask):
        """
        Runs the encoder only.
        Args:
            src: [batch, src_len]
            src_mask: [batch, src_len] (Padding mask)
        Returns:
            Tensor: Encoded memory [batch, src_len, d_model]
        """
        # Apply embedding + positional encoding
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.d_model))
        
        # Pass through the Transformer Encoder
        # Note: src_key_padding_mask expects True for positions to ignore
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_mask)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        """
        Runs the decoder only.
        Args:
            tgt: [batch, tgt_len] - The target sequence generated so far
            memory: [batch, src_len, d_model] - Output from the encoder
            src_mask: [batch, src_len] - Mask for the source (memory)
            tgt_mask: [tgt_len, tgt_len] - Causal mask for the target
        Returns:
            Tensor: Logits [batch, tgt_len, vocab_size]
        """
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model))
        
        # Create padding mask for target to ignore padding tokens (if any)
        tgt_key_padding_mask = (tgt == self.pad_idx)
        
        # Pass through the Transformer Decoder
        out = self.transformer.decoder(
            tgt_emb, memory, 
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_mask # Important: Mask padding in the memory (Cross-Attention)
        )
        return self.generator(out)

    def forward(self, src, tgt):
        """
        Standard forward pass for training.
        Automatically handles mask creation and calls encode/decode internally.
        """
        # Generate Masks
        tgt_mask = self.make_tgt_mask(tgt)
        src_mask = self.make_src_mask(src)
        
        # Encode
        memory = self.encode(src, src_mask)
        
        # Decode
        logits = self.decode(tgt, memory, src_mask, tgt_mask)
        
        return logits