import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Callable, Optional
from torch import Tensor

class CausalConv(nn.Module):
    # nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=depthwise_conv, padding=1,
    #                                         groups=d_model, bias=bias)
    def __init__(self, c_in, c_out, kernel_size, **kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, **kwargs)

    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        # Pad left side with kernel_size-1 zeros
        padding = self.kernel_size - 1
        x = F.pad(x, (padding, 0))  # pad only left side
        return self.conv(x)

class DecoderOnlyTransformerLayer(nn.Module):
    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = nn.functional.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None, depthwise_conv=3) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                               bias=bias, **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Depth-wise convolution over 3 timesteps
        if depthwise_conv > 0:
            # transform
            # nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=depthwise_conv, padding=1,
            #                                         groups=d_model, bias=bias)
            # into causal conv:
            self.depthwise_conv = CausalConv(c_in=d_model, c_out=d_model, kernel_size=depthwise_conv, groups=d_model, bias=bias)
        else:
            self.depthwise_conv = None


        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = self._get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = nn.functional.relu
        super().__setstate__(state)

    def forward(self, tgt: Tensor, tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_is_causal: bool = False) -> Tensor:
        x = tgt

        if self.depthwise_conv is not None:
            # Reshape and apply depth-wise convolution
            # Input tensor has shape (batch_size, seq_len, d_model),
            # Conv1d expects (batch_size, d_model, seq_len), so we need to transpose
            x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
            x = self.depthwise_conv(x)
            x = x.transpose(1, 2)  # Back to (batch_size, seq_len, d_model)

        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return nn.functional.relu
        elif activation == "gelu":
            return nn.functional.gelu
        raise RuntimeError(f"activation should be relu/gelu, not {activation}")


class TransformerBlock(nn.Module):
    """
    Decoder-only Transformer layers for causal language modeling
    with optional input reshaping
    """
    support_streaming = False

    def __init__(self, dimension_e, nhead=8, num_layers=6, feedforward_mult=None,
                 activation=None, input_to_latent=None, dropout=0, depthwise_conv=3):
        super().__init__()

        if feedforward_mult is None:
            feedforward_mult = 3
        dim_feedforward = int(feedforward_mult * dimension_e)

        self.input_to_latent = input_to_latent

        self.layers = nn.ModuleList([
            DecoderOnlyTransformerLayer(
                d_model=dimension_e,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=activation or 'gelu',
                batch_first=True,
                norm_first=True,
                bias=False,
                dropout=dropout,
                depthwise_conv=depthwise_conv,
            ) for _ in range(num_layers)
        ])

        self.causal_mask = None

    def forward(self, x):
        # Apply input_to_latent if provided
        if self.input_to_latent is not None:
            x = self.input_to_latent(x)

        B,T,C,H = x.size()
        x = x.permute(0,2,1,3)
        x = x.reshape(B*C, T, H)

        x_original = x.clone()



        # x shape after potential reshaping: (batch_size, sequence_length, dimension_e)
        if self.causal_mask is None or self.causal_mask.size(0) != x.size(1):
            self.causal_mask = self._generate_square_subsequent_mask(x.size(1)).to(x.device)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, tgt_mask=self.causal_mask, tgt_is_causal=True)


        return x, x_original

    @staticmethod
    def _generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
class Simple_channel_mixing_module(nn.Module):
    def __init__(self, H, C):
        super().__init__()
        self.H = H
        self.C = C

        # Learnable H×H matrices
        self.R = nn.Parameter(torch.randn(H, H)) #channel with itself
        self.Q = nn.Parameter(torch.randn(H, H)) #channel with other channels

        # to build the matrix with R on the diagonal and Q elsewhere
        I = torch.eye(C)
        mask = torch.ones(C, C) - I

        # Register them as buffers so they move with .to(device) but aren't parameters
        self.register_buffer('I', I)
        self.register_buffer('mask', mask)

    def forward(self, x):
        # x:    [B, T, C*H]
        # I:       [C, C]
        # mask:    [C, C]
        # R, Q:    [H, H]
        # kron(I, R) → [H*C, H*C], puts R on diagonal blocks
        # kron(mask, Q) → [H*C, H*C], fills off‑diag with Q
        M = torch.kron(self.I, self.R) + torch.kron(self.mask, self.Q)
        x_hat = x.matmul(M.T)

        return x_hat