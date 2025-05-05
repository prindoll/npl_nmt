import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional
import math
import logging

# class PositionalEncoder(nn.Module):
#     def __init__(self, d_model, max_seq_length=200, dropout=0.1):
#         super().__init__()
        
#         self.d_model = d_model
#         self.dropout = nn.Dropout(dropout)
#         self._max_seq_length = max_seq_length
        
#         pe = torch.zeros(max_seq_length, d_model)
        
#         for pos in range(max_seq_length):
#             for i in range(0, d_model, 2):
#                 pe[pos, i] = math.sin(pos/(10000**(2*i/d_model)))
#                 pe[pos, i+1] = math.cos(pos/(10000**((2*i+1)/d_model)))
#         pe = pe.unsqueeze(0)        
#         self.register_buffer('pe', pe)

#         @torch.jit.script
#         def splice_by_size(source, target):
#             """Custom function to splice the source by target's second dimension. Required due to torch.Size not a torchTensor. Why? hell if I know."""
#             length = target.size(1);
#             return source[:, :length]

#         self.splice_by_size = splice_by_size
#     def forward(self, x):
#         if(x.shape[1] > self._max_seq_length):
#             logging.warn("Input longer than maximum supported length for PE detected. Build a model with a larger input_max_length limit if you want to keep the input; or ignore if you want the input trimmed")
#             x = x[:, x:self._max_seq_length]
        
#         x = x * math.sqrt(self.d_model)
        
#         spliced_pe = self.splice_by_size(self.pe, x) # self.pe[:, :x.shape[1]]
# #        pe = Variable(spliced_pe, requires_grad=False)
#         pe = spliced_pe.requires_grad_(False)
        
# #        if x.is_cuda: # remove since it is a sub nn.Module
# #            pe.cuda()
# #        assert all([xs == ys for xs, ys in zip(x.shape[1:], pe.shape[1:])]), "{} - {}".format(x.shape, pe.shape)

#         x = x + pe
#         x = self.dropout(x)
        
#         return x

class RotaryPositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_length=200, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self._max_seq_length = max_seq_length
        
        # Ensure d_model is even for RoPE (required for splitting into pairs)
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        
        # Precompute frequency terms for RoPE
        self._freqs = self._compute_freqs(d_model, max_seq_length)
        self.register_buffer('freqs', self._freqs)

    def _compute_freqs(self, d_model, max_seq_length):
        """Compute the frequency terms for RoPE."""
        theta = 10000.0 ** (-torch.arange(0, d_model, 2) / d_model)
        positions = torch.arange(max_seq_length).unsqueeze(1)
        freqs = positions * theta.unsqueeze(0)
        return freqs

    def _apply_rotary_embedding(self, x):
        """Apply rotary embeddings to the input tensor."""
        seq_len = x.size(1)
        freqs = self.freqs[:seq_len].to(x.device)
        
        # Split x into pairs for rotation
        x_even = x[..., 0::2]  # Even-indexed dimensions
        x_odd = x[..., 1::2]   # Odd-indexed dimensions
        
        # Compute angles and apply rotation
        angles = freqs
        cos_theta = torch.cos(angles)
        sin_theta = torch.sin(angles)
        
        # Rotate: x_even' = x_even * cos - x_odd * sin
        #         x_odd'  = x_even * sin + x_odd * cos
        x_even_rot = x_even * cos_theta - x_odd * sin_theta
        x_odd_rot = x_even * sin_theta + x_odd * cos_theta
        
        # Interleave the rotated pairs back
        x_rot = torch.stack([x_even_rot, x_odd_rot], dim=-1)
        x_rot = x_rot.view(*x.shape)
        
        return x_rot

    def forward(self, x):
        # Check for sequence length
        if x.shape[1] > self._max_seq_length:
            logging.warn("Input longer than maximum supported length for RoPE detected. Trimming input.")
            x = x[:, :self._max_seq_length]
        
        # Scale input embeddings
        x = x * math.sqrt(self.d_model)
        
        # Apply rotary embeddings
        x = self._apply_rotary_embedding(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        # three casting linear layer for query/key.value
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        """
        Args:
            q / k / v: query/key/value, should all be [batch_size, sequence_length, d_model]. Only differ in decode attention, where q is tgt_len and k/v is src_len
            mask: either [batch_size, 1, src_len] or [batch_size, tgt_len, tgt_len]. The last two dimensions must match or are broadcastable.
        Returns:
            the value of the attention process, [batch_size, sequence_length, d_model].
            The used attention, [batch_size, q_length, k_v_length]
        """
        bs = q.shape[0]
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        value, attn = self.attention(q, k, v, mask, self.dropout)
        concat = value.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output, attn

    def attention(self, q, k, v, mask=None, dropout=None):
        """Calculate the attention and output the attention & value
        Args:
            q / k / v: query/key/value already transformed, should all be [batch_size, heads, sequence_length, d_k]. Only differ in decode attention, where q is tgt_len and k/v is src_len
            mask: either [batch_size, 1, src_len] or [batch_size, tgt_len, tgt_len]. The last two dimensions must match or are broadcastable.
        Returns: 
            the attentionized but raw values [batch_size, head, seq_length, d_k]
            the attention calculated [batch_size, heads, sequence_length, sequence_length]
        """
    
#        d_k = q.shape[-1]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1) # add a dimension to account for head
            scores = scores.masked_fill(mask==0, -1e9)
        # softmax the padding/peeking masked attention
        scores = functional.softmax(scores, dim=-1)
        
        if dropout is not None:
            scores = dropout(scores)
        
        output = torch.matmul(scores, v)
        return output, scores

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class FeedForward(nn.Module):
    """A two-hidden-linear feedforward layer that can activate and dropout its transition state"""
    def __init__(self, d_model, d_ff=2048, internal_activation=functional.relu, dropout=0.1):
        super().__init__() 
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

        self.internal_activation = internal_activation
    
    def forward(self, x):
        x = self.dropout(self.internal_activation(self.linear_1(x)))
        x = self.linear_2(x)
        return x
