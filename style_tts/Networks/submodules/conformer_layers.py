import math
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from .basic_layers import RelPositionalEncoding, Conv2dSubsampling, PositionwiseFeedForward, RelPositionMultiHeadedAttention, ConvolutionModule, LayerNorm, MultiEmbedding, StyleMixingLayer

class ConformerEncoder(nn.Module):
    """Conformer encoder module.
    Args:
        input_size (int): Input dimension.
        output_size (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        attention_dropout_rate (float): Dropout rate in attention.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
            If True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            If False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        rel_pos_type (str): Whether to use the latest relative positional encoding or
            the legacy one. The legacy relative positional encoding will be deprecated
            in the future. More Details can be found in
            https://github.com/espnet/espnet/pull/2816.
        encoder_pos_enc_layer_type (str): Encoder positional encoding layer type.
        encoder_attn_layer_type (str): Encoder attention layer type.
        activation_type (str): Encoder activation function type.
        use_cnn_module (bool): Whether to use convolution module.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.
        cnn_module_kernel (int): Kernerl size of convolution module.
        padding_idx (int): Padding idx for input_layer=embed.
    """

    def __init__(
            self,
            dim_in: int = 49,
            dim_hidden: int = 256,
            num_heads: int = 4,
            dim_linear_hidden: int = 2048,
            num_blocks: int = 6,
            p_dropout: float = 0.1,
            p_positional_dropout: float = 0.1,
            p_attention_dropout: float = 0.0,
            positionwise_conv_kernel_size: int = 3,
            use_cnn_module: bool = True,
            zero_triu: bool = False,
            cnn_module_kernel: int = 31,
            padding_idx: int = -1,
            interctc_layer_idx: List[int] = [],
            embedding_type="convsubsampling",
            num_channels=1,
    ):
        super().__init__()
        self._output_size = dim_hidden
        activ = nn.SiLU()
        pos_enc_class = RelPositionalEncoding
        self.pos_enc = pos_enc_class(dim_hidden, p_positional_dropout)
        if embedding_type == "convsubsampling":
            self.embed = Conv2dSubsampling(
                dim_in,
                dim_hidden,
                p_dropout,
            )
        elif embedding_type == "multilabel":
            self.embed = MultiEmbedding(dim_in, dim_hidden, p_dropout, num_channels=num_channels)
        else:
            self.embed = nn.Embedding(dim_in, dim_hidden)

        positionwise_layer= PositionwiseFeedForward
        positionwise_layer_kwargs = dict(
            dim_in=dim_hidden, dim_hidden=dim_linear_hidden, p_dropout=p_dropout, activ=activ)

        encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_kwargs = dict(
            n_head=num_heads, n_feat=dim_hidden, dropout_rate=p_attention_dropout, zero_triu=zero_triu)

        convolution_layer = ConvolutionModule
        convolution_layer_kwargs = dict(
            channels=dim_hidden, kernel_size=cnn_module_kernel, activation=activ)

        self.encoders = nn.ModuleList(
            [EncoderLayer(
                size=dim_hidden,
                self_attn=encoder_selfattn_layer(**encoder_selfattn_layer_kwargs),
                feed_forward=positionwise_layer(**positionwise_layer_kwargs),
                conv_module=convolution_layer(**convolution_layer_kwargs),
                dropout_rate=p_dropout) for _ in range(num_blocks)])

        self.after_norm = LayerNorm(dim_hidden)
        self.conditioning_layer = None
        self.embedding_type = embedding_type

    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (#batch, L, input_size).
            mask (Tensor): Input mask (#batch, 1, L). 0 is masked, 1 is unmasked
        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
        """
        assert (x.size(1) >= 7) or (self.embedding_type != "convsubsampling")
        if isinstance(self.embed, nn.Embedding):
            x = self.embed(x)
        else:
            x, mask = self.embed(x, mask)

        x = self.pos_enc(x)
        intermediate_outs = []
        for layer in self.encoders:
            x, mask = layer(x, mask)

        if isinstance(x, tuple):
            x = x[0]

        x = self.after_norm(x)
        if len(intermediate_outs) > 0:
            return (x, intermediate_outs), mask
        return x, mask


class EncoderLayer(nn.Module):
    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        conv_module,
        dropout_rate,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.conv_module = conv_module
        self.norm_ff = LayerNorm(size)  # for the FNN module
        self.norm_mha = LayerNorm(size)  # for the MHA module
        self.norm_ff_pre = LayerNorm(size)
        self.ff_scale = 0.5
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(size)  # for the CNN module
            self.norm_final = LayerNorm(size)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size

    def forward(self, x_input, mask, cache=None):
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        residual = x
        x = self.norm_ff_pre(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))

        # multi-headed self-attention module
        residual = x
        x = self.norm_mha(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if pos_emb is not None:
            x_att = self.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x_q, x, x, mask)
        x = residual + self.dropout(x_att)

        # convolution module
        residual = x
        x = self.norm_conv(x)
        x = residual + self.dropout(self.conv_module(x)) * (1 - mask.transpose(1, 2)) # add mask on conv

        # feed forward module
        residual = x
        x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))

        if self.conv_module is not None:
            x = self.norm_final(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask
