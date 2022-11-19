import torch
from torch import nn
import torch.nn.functional as F
from .submodules.conformer_layers import ConformerEncoder

class TextEncoder(nn.Module):
    def __init__(self, encocer_config={}):
        super().__init__()
        self.phoneme_encoder = ConformerEncoder(**encoder_config)

    def forward(self, text, text_mask):
        enc, enc_mask = self.phoneme_encoder(text, text_mask)
        return enc, enc_mask


class ASRPredictor(nn.Module):
    def __init__(self, encocer_config={}, decoder_config={}):
        super().__init__()
        self.mel_encoder = ConformerEncoder(**encoder_config)
        self.phoneme_decoder = CumsumAttentionDecoder(**decoder_config)

    def forward(self, text, text_mask, mel, mel_mask):
        enc, enc_mask = self.mel_encoder(mel, mel_mask)
        dec = self.phoneme_decoder(text, text_mask, enc, enc_mask)
        return enc, dec


class MelDecoder(nn.Module):
    def __init__(self, decoder_config={}):
        super().__init__()
        self.mel_decoder = ConformerDecoder(**decoder_config)

    def forward(self, enc, enc_mask, spk_id=None, style_vec=None):
        dec = self.mel_decoder(enc, enc_mask, spk_id, style_vec)
        return dec


class PitchEncoder(nn.Module):
    def __init__(self, encocer_config={}, decocer_config={}):
        super().__init__()
        self.pitch_decoder = ConformerDecoder(**decoder_config)
        self.pitch_encoder = nn.Embedding(**encoder_config)

    def forward(self, enc, enc_mask):
        dec, dec_mask = self.pitch_encoder(enc, enc_mask)
        return dec, dec_mask

    def encode(self, dec, dec_mask):
        enc = self.pitch_encoder(dec)
        return enc


class EnergyEncoder(nn.Module):
    def __init__(self, encocer_config={}, decocer_config={}):
        super().__init__()
        self.energy_decoder = ConformerDecoder(**decoder_config)
        self.energy_encoder = nn.Embedding(**encoder_config)

    def forward(self, enc, enc_mask):
        dec, dec_mask = self.energy_encoder(enc, enc_mask)
        return dec, dec_mask

    def encode(self, dec, dec_mask):
        enc = self.energy_encoder(dec)
        return enc


class DurationPredictor(nn.Module):
    def __init__(self, config={}):
        super().__init__()
        self.predictor = ConformerDecoder(**config)

    def forward(self, enc, enc_mask):
        pred = self.predictor(enc, enc_mask)
        return pred


class RandomAlignmentModule(nn.Module):
    def __init__(self, soft_aligner_config={}, hard_aligner_config={}, p_soft=0.5):
        super().__init__()
        self.soft_aligner = SoftAligner(**config)
        self.hard_aligner = HardAligner(**config)
        self.p_soft = p_soft

    def forward(self, text_enc, text_mask, mel_enc, mel_mask):
        if np.random.random() < self.p_soft:
            alignment = self.soft_aligner(text_enc, text_mask, mel_enc, mel_mask)
        else:
            alignment = self.hard_aligner(text_enc, text_mask, mel_enc, mel_mask)
        return alignment


class LabelAdaptor(nn.Module):
    def __init__(self, emb_config={}, adaptor_config={}):
        super().__init__()
        self.label_encoder = nn.Embedding(**emb_config)
        self.adaptor = Adaptor(**adaptor_config)

    def forward(self, x, label):
        label_enc = self.label_encoder(label)
        x = self.adaptor(x, label_enc)
        return x


class VecoderAdaptor(nn.Module):
    def __init__(self, adaptor_config={}):
        super().__init__()
        self.adaptor = Adaptor(**adaptor_config)

    def forward(self, x, vec):
        x = self.adaptor(x, vec)
        return x

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

