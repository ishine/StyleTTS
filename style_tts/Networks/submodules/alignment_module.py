import torch
from torch import nn
import torch.nn.functional as F

class Aligner(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, h_text, alignment=None, durations=None):
        """
        Args:
        - h_text (FloatTensor): size=(B, Lt, Ht)
        - alignment (FloatTensor): size=(B, Lm, Lt)
        - duration (LongTensor): size=(B, Lt), sum(duration)=Lm
        """
        if alignment is not None:
            h_text_up = torch.bmm(alignment, h_text)
        else:
            h_text_up = self._duration_upsample(h_text, durations)

        return h_text_up

    def _duration_upsample(self, xs, ds):
        repeated_xs = [torch.repeat_interleave(x, d, dim=0) for x, d in zip(xs, ds)]
        return torch.nn.utils.rnn.pad_sequence(repeated_xs, batch_first=True)


