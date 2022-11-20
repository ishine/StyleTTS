import torch
from torch import nn
import torch.nn.functional as F


class MelToVariance(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConformerEncoder()
        self.to_pitch = nn.Sequential(
            nn.Conv1d(dim_hidden, dim_hidden*2, 5, 1, 2),
            nn.GLU(dim=1),
            nn.Linear(dim_hidden, 1)
        )
        self.to_energy = nn.Sequential(
            nn.Conv1d(dim_hidden, dim_hidden*2, 5, 1, 2),
            nn.GLU(dim=1),
            nn.Linear(dim_hidden, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        p = self.to_pitch(x)
        e = self.to_energy(x)
        return p, e

class MaskedLayerNorm1d(nn.Module):
    def __init__(self, dim_hidden, gamma0: float=0.1):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.gamma = torch.nn.Parameter(torch.ones(1, dim_hidden, 1) * gamma0)
        self.beta = torch.nn.Parameter(torch.zeros(1, dim_hidden, 1))

    def forward(self, x, x_mask):
        """
        x.size = (B, C, L)
        x_mask = (B, 1, L), 1 is masked
        """
        dtype = x.dtype
        max_length = x.size(2)
        x = x.float()
        x_mask = x_mask.float()
        x = x * (1 - x_mask)
        _length = (1 - x_mask).sum(dim=2, keepdim=True)
        x_mean = (x / _length).mean(dim=1, keepdim=True).sum(dim=2, keepdim=True)
        x_centered = (x - x_mean) * (1 - x_mask)
        x_std = torch.sqrt(
            1e-5 + (x_centered**2 / _length).mean(dim=1, keepdim=True).sum(dim=2, keepdim=True)
        )
        x = ((x - x_mean) / (1e-5 + x_std)) * (1 - x_mask)
        x = x.type(dtype)
        x_mask = x_mask.type(dtype)
        x = x * self.gamma + self.beta
        return x, x_mask

class MaskedConvLN1d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, p_dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(
            dim_in, dim_out*2, kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.glu = nn.GLU(dim=1)
        self.ln = MaskedLayerNorm1d(dim_out)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv(x * (1 - x_mask))
        x = self.glu(x)
        x, x_mask = self.ln(x, x_mask)
        x = self.drop(x)
        return x, x_mask

class VarianceEncoder(nn.Module):
    def __init__(self, n_layers=4, dim_hidden=256, dim_out=1, kernel_size=5, dropout_rate=0.2):
        super().__init__()
        self.convs = nn.ModuleList([
            MaskedConvLN1d(dim_hidden, dim_hidden, kernel_size, dropout_rate) for _ in range(n_layers)
        ])
        self.last_linear = nn.Linear(dim_hidden, dim_out)

    def forward(self, x, x_mask):
        x = x.transpose(1, 2)
        for conv in self.convs:
            x, _ = conv(x, x_mask)
        x = self.last_linear(x.transpose(1, 2))
        return x


### ---------- Variance Adaptors -----------
class AddAdaptor(nn.Module)
    def __init__(self):
        super().__init__()

    def forward(self, x, sty):
        x = x + sty
        return x

class StyleAdaptor(nn.Module)
    def __init__(self, dim_sty, dim_hid):
        super().__init__()
        self.lin_s1 = nn.Linear(dim_sty, dim_hid*2)
        self.lin_h1 = nn.Linear(dim_hid, dim_hid*2)
        self.lin_2 = nn.Linear(dim_hid, dim_hid*2)
        self.glu = nn.GLU(dim=-1)

    def forward(self, x, sty):
        res = x
        s = self.lin_s1(sty)
        x = self.lin_h1(x)
        x = self.glu(x + s)
        x = (res + x) / math.sqrt(2)
        res2 = x
        x = self.lin_2(x)
        x = self.glu(x)
        x = (res2 + x) /math.sqrt(2)
        return x

class LabelAdaptor(nn.Module):
    def __init__(self, emb_config={}, adaptor_config={}):
        super().__init__()
        self.label_encoder = nn.Embedding(**emb_config)
        self.adaptor = StyleAdaptor(**adaptor_config)

    def forward(self, x, label):
        label_enc = self.label_encoder(label)
        x = self.adaptor(x, label_enc)
        return x

class VectorAdaptor(nn.Module):
    def __init__(self, adaptor_class):
        super().__init__()
        self.adaptor = eval(adaptor_class)(**adaptor_config)

    def forward(self, x, vec):
        x = self.adaptor(x, vec)
        return x


