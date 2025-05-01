import torch
import torch.nn as nn
from lstm import LSTM

class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, dropout=0.1, device=None):
        super().__init__()
        
        # LSTM layer
        self.lstm = LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            device=device,
        )
        
        # Calculate the size of the output from LSTM
        # lstm_output_dim = cfg['hidden_dim'] * 2 if cfg['bidirectional'] else cfg['hidden_dim']
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # self.norm = nn.LayerNorm(lstm_output_dim)
        
    def forward(self, x):
        
        # Pass through LSTM
        output, (hidden, cell) = self.lstm(x)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Pass through layer normalization
        return output

class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, emb_dim*2),
            nn.ReLU(),
            nn.Linear(emb_dim*2, emb_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads=1, dropout=0.0, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # reshape x so that each emb divided by num_heads
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # reshape from (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(
            mask_bool, -torch.inf
        )

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5,
            dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        # reshape from (b, num_heads, num_tokens, head_dim) to (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec

class ALSTMModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_directions = 2 if cfg['bidirectional'] else 1
        self.hidden_size = cfg['hidden_size'] * self.num_directions
        self.embedding = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'], padding_idx=0)
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        self.lstm = LSTMBlock(
            input_size=cfg['emb_dim'],
            hidden_size=cfg['hidden_size'],
            num_layers=cfg['lstm_layers'],
            bidirectional=cfg['bidirectional'],
            dropout=cfg['drop_rate'],
            device=cfg['device']
        )
        self.attn_forward = MultiHeadAttention(
            d_in=cfg['hidden_size'],
            d_out=cfg['hidden_size'],
            context_length=cfg['context_length'],
            num_heads=cfg['n_heads'],
            dropout=cfg['drop_rate'],
            qkv_bias=cfg['qkv_bias']
        )
        
        self.attn_backward = MultiHeadAttention(
            d_in=cfg['hidden_size'],
            d_out=cfg['hidden_size'],
            context_length=cfg['context_length'],
            num_heads=cfg['n_heads'],
            dropout=cfg['drop_rate'],
            qkv_bias=cfg['qkv_bias']
        )
         

        self.norm2 = nn.LayerNorm(self.hidden_size)
        self.ff = FeedForward(self.hidden_size)
        self.output = nn.Linear(self.hidden_size, cfg['output'])
    
    def forward(self, in_idx):
        x = self.embedding(in_idx)
        x = self.drop_emb(x)
        x = self.lstm(x)
        x_forward, x_backward = torch.chunk(x, 2, dim=-1)

        x_forward = self.attn_forward(x_forward)
        x_backward = self.attn_backward(x_backward)

        x = torch.concat((x_forward, x_backward), dim=-1)
        x = self.norm2(x)
        x = self.ff(x)
        logits = self.output(x)
        return logits[:, -1, :]