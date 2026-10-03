import math
import torch
import torch.nn as nn
from typing import Tuple

from .layers import PositionalEncoding, EncoderBlock, DecoderBlock

class Transformer(nn.Module):
    """
    Original Transformer architecture (Encoder + Decoder) as described in 
    'Attention Is All You Need' (Vaswani et al., 2017).
    """
    def __init__(
        self, 
        src_vocab_size: int, 
        tgt_vocab_size: int, 
        d_model: int = 512, 
        num_heads: int = 8, 
        num_layers: int = 6, 
        d_ff: int = 2048, 
        max_len: int = 5000, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.encoder_layers = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        tgt_subsequent_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_subsequent_mask
        return src_mask, tgt_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # Encoder
        enc_output = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
            
        # Decoder
        dec_output = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
            
        return self.fc_out(dec_output)
