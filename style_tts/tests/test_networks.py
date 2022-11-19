"""
テスト内容

1. text encoder に mask が機能しているか
2. mel decoder に mask が機能しているか
3. asr decoder に mask が機能しているか
4. asr decoder の attention に mask が機能しているか
5. length regulator が機能しているか
6. length regulator に mask が機能しているか
7. variation adaptor に mask が機能しているか
8. loss function に mask が機能しているか
"""
import unittest
import torch
from styletts.Networks.modules import TextEncoder

class TestNetworks(unittest.TestCase):

    def test_textencoder(self):
        text_encoder_config = dict(
            dim_in=49, # ntokens,
            cnn_module_kernel=7,
            dim_hidden=512,
            embedding_type="multilabel",
            num_channels=2,
        )
        encoder = TextEncoder(text_encoder_config)
        x = torch.randint(0,49, (2, 10, 2))
        m = (torch.arange(10) < 8).float().reshape(1, 1, -1)
        x = torch.randint(0,49, (2, 10, 2))
        _ = te.eval()
        with torch.no_grad():
            z1 = te(x, m)
            x[:, -1] = 0
            z2 = te(x, m)
        self.assertTrue((z1[0] - z2[0]).abs().sum() < 1e-3)

