import torch
from torch import nn
import torch.nn.functionals as F

class StyleTTS(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        pass

class SoftAligedConformerTTS(nn.Module):
    """
    Training ConformerTTS with aligment rules of StyleTTS
    """
    def __init__(
            self,
            text_config={},
            mel_decoder_config={},
            asr_encoder_config={},
            pitch_encoder_config={},
            energy_encoder_config={},
            alignment_config={},
            speaker_encoder_config={}
    ):
        super().__init__()
        self.text_encoder = TextEncoder(**text_encoder_config)
        self.mel_decoder = MelDecoder(**mel_decoder_config)
        self.asr_encoder = ASREncoder(**asr_encoder_config)
        self.pitch_encoder = PitchEncoder(**pitch_encoder_config)
        self.energy_encoder = EnergyEncoder(**energy_encoder_config)
        self.spk_adaptor= LabelAdaptor(**spk_encoder_config)
        self.style_adaptor = VectorAdaptor(**style_encoder_config)
        self.duration_predictor = DurationPredictor(**duration_predictor_config)
        self.aligner = AlignmentModule(**alignment_config)

    def forward(
            self,
            text,
            text_mask,
            mel,
            mel_mask,
            style_vec,
            spk_id=None,
            pitch=None,
            energy=None,
    ):
        text_enc = self.text_encoder(text, text_mask)
        mel_enc, asr_pred = self.asr_encoder(mel, mel_mask)
        alignment = self.aligner(text_enc, text_mask, mel_enc, mel_mask)
        text_enc, pitch_pred, energy_pred = self._adaptive(text_enc, spk_id, style_vec, pitch, energy)
        mel_pred = self.mel_decoder(text_enc, alignment, mel_mask, spk_id, style_vec)
        return {
            "mel_pred": mel_pred,
            "alignment": alignment,
            "asr_pred": asr_pred,
            "pitch_pred": pitch_pred,
            "energy_pred": energy_pred,
        }

    def inference(self, text, text_mask):
        pass

    def _adaptive(self, text_enc, spk_id=None, style_vec=None, pitch=None, energy=None):
        """
        TODO: comparing adaptation of pitch and energy on text domain or mel domain
        """
        x = text_enc
        if spk_id is not None:
            x = self.spk_adaptor(x, spk_id)
        if style_vec is not None:
            x = self.style_adaptor(x, style_vec)
        if pitch is not None:
            pitch = self.pitch_predictor(x)
        pitch = self.pitch_predictor.encode(pitch)
        if energy is not None:
            energy = self.energy_predictor(x)
        energy = self.energy_predictor.encode(pitch)
        x = x + pitch + energy
        return x
