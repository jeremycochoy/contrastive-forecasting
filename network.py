import torch.nn as nn
import torch.nn.functional as F
from blocks import TransformerBlock, Simple_channel_mixing_module

class Simple_encoder(nn.Module): #encoder proposed during our meeting
    def __init__(self, input_shape, intermediate_dim, H, dropout_p=0.0):

        W = input_shape[-1]

        super().__init__()

        self.linear1 = nn.Linear(W, intermediate_dim)
        self.linear2 = nn.Linear(intermediate_dim, H)
        self.dropout = nn.Dropout(p=dropout_p)

        self.linear_skipping = nn.Linear(W, H)
        self.layer_norm = nn.LayerNorm(H)

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = F.relu(x1)
        x1 = self.linear2(x1)

        x2 = self.linear_skipping(x)

        return self.layer_norm(x1+x2)


class SimpleModel(nn.Module):
    def __init__(self, C, H, W, intermediate_dim=64, num_layers=4, nhead=4, feedforward_mult=2, dropout=0.1):
        """
        Args:
            C: channels
            H: output dim of encoder
            W: raw input dimension per channel
            intermediate_dim: intermediate dim in the encoder
            num_layers: number of transformer layers
            nhead: number of attention heads
            feedforward_mult: multiplier for FFN hidden dim in transformer
            dropout: dropout rate
        """
        super().__init__()

        self.C = C
        self.H = H
        self.W = W

        # One encoder shared across channels (assumes input shape [B, T, C, W])
        self.encoder = Simple_encoder(input_shape=(None, None, C, W), intermediate_dim=intermediate_dim, H=H)

        # Transformer expects input of shape [B*C, T, H]
        self.transformer = TransformerBlock(
            dimension_e=H,
            nhead=nhead,
            num_layers=num_layers,
            feedforward_mult=feedforward_mult,
            dropout=dropout,
            input_to_latent=self.encoder
        )

        # Channel mixing module expects shape [B, T, C*H]
        self.channel_mixing_module = Simple_channel_mixing_module(H=H, C=C)

    def forward(self, x):
        #x  [B, T_raw, C]
        #split into windows
        B, T_raw, C = x.shape
        W = self.W
        H = self.H

        assert T_raw % W == 0, "T_raw must be divisible by window size W"

        T = T_raw // W
        x = x.view(B, T, W, C)          # [B, T, W, C]
        x = x.permute(0, 1, 3, 2)       # [B, T, C, W]

        x, x_original = self.transformer(x)

        x = x.reshape(B,C,T,H)
        x = x.permute(0,2,1,3)
        x = x.reshape(B,T,C*H)

        x_original = x_original.reshape(B,C,T,H)
        x_original = x_original.permute(0,2,1,3)

        x = self.channel_mixing_module(x)

        x = x.reshape(B,T,C,H)

        return x, x_original

