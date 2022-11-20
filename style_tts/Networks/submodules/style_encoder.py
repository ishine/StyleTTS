import math
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio

class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'half':
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError(
                'Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)

class GroupNormFP32(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            weight=self.weight.float() if self.weight is not None else None,
            bias=self.bias.float() if self.bias is not None else None,
            eps=self.eps,
        )
        return output.type_as(input)

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 norm_type=None, downsample='none'):
        super().__init__()
        self.actv = actv
        self.norm_type = norm_type
        self.downsample = DownSample(downsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.norm_type == "instance":
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        elif self.norm_type == "layer":
            self.norm1 = GroupNormFP32(1, dim_in, affine=True)
            self.norm2 = GroupNormFP32(1, dim_in, affine=True)
        elif self.norm_type is None:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample(x)
        x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class StyleEncoder(nn.Module):
    def __init__(self, dim_in=48, style_dim=128, max_conv_dim=384):
        super().__init__()
        blocks = []
        blocks += [nn.Conv2d(1, dim_in, 3, 1, 1)]

        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, norm_type="layer", downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 5, 1, 0)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)
        self.last = nn.Conv2d(dim_out, style_dim, 1)
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_fft=2048, n_mels=80, win_length=1200, hop_length=300,
            pad=(2048 - 300)//2, center=False,
            f_max=8000, f_min=20, mel_scale="slaney", power=1, norm="slaney")
        self.to_mel.requires_grad = False
        self.mel_mean, self.mel_std = -5, 5

    def forward(self, x):
        with torch.no_grad():
            x = torch.log(1e-5 + self.to_mel(x))
            x = (x - self.mel_mean) / self.mel_std
        x = x.unsqueeze(1)
        h = self.shared(x)
        s = self.last(h).squeeze(3).squeeze(2)
        return s
