import torch
from torch import nn
import torch.nn.functional as F
import math

class CumAttentionDecoder(nn.Module):
    def __init__(self, dim_emb=256,  dim_hidden=512, n_loc_filters=32, loc_kernel_size=63,
                 n_tokens=49, input_channels=1):
        super().__init__()
        self.pad_token = 0
        self.sos_token = 1
        self.eos_token = 2
        self.unk_token = 3
        self.random_mask = 0.1
        self.embedding = nn.Embedding(n_tokens, dim_emb)
        self.decoder_rnn_dim = dim_hidden
        self.project_to_n_symbols = nn.Linear(self.decoder_rnn_dim, n_tokens)
        self.attention_layer = Attention(
            self.decoder_rnn_dim,
            dim_hidden,
            dim_hidden,
            n_loc_filters,
            loc_kernel_size
        )
        self.decoder_rnn = nn.LSTMCell(self.decoder_rnn_dim + dim_emb, self.decoder_rnn_dim)
        self.project_to_hidden = nn.Sequential(
            nn.Linear(self.decoder_rnn_dim * 2, dim_hidden), nn.Tanh())

    def initialize_decoder_states(self, memory, mask):
        """
        moemory.shape = (B, L, H) = (Batchsize, Maxtimestep, Hiddendim)
        """
        B, L, H = memory.shape
        self.decoder_hidden = torch.zeros((B, self.decoder_rnn_dim)).type_as(memory)
        self.decoder_cell = torch.zeros((B, self.decoder_rnn_dim)).type_as(memory)
        self.attention_weights = torch.zeros((B, L)).type_as(memory)
        self.attention_weights_cum = torch.zeros((B, L)).type_as(memory)
        self.attention_context = torch.zeros((B, H)).type_as(memory)
        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def forward(self, memory, memory_mask, text_input, use_start_token=False):
        """
        moemory.shape = (B, L, H)
        moemory_mask.shape = (B, L, )
        texts_input.shape = (B, T)
        """
        self.initialize_decoder_states(memory, memory_mask)
        # text random mask
        random_mask = (torch.rand(text_input.shape) < self.random_mask).to(text_input.device)
        _text_input = text_input.clone()
        _text_input.masked_fill_(random_mask, self.unk_token)
        decoder_inputs = self.embedding(_text_input).transpose(0, 1) # -> [T, B, channel]
        if use_start_token:
            start_embedding = self.embedding(
                torch.LongTensor([self.sos_token]*decoder_inputs.size(1)).to(decoder_inputs.device))
        else:
            start_embedding = self.embedding(text_input[:, :1])
            text_input = text_input[:, 1:]

        decoder_inputs = torch.cat((start_embedding.unsqueeze(0), decoder_inputs), dim=0)
        hidden_outputs, logit_outputs, alignments = [], [], []
        while len(hidden_outputs) < decoder_inputs.size(0):
            decoder_input = decoder_inputs[len(hidden_outputs)]
            hidden, logit, attention_weights = self.decode(decoder_input)
            hidden_outputs += [hidden]
            logit_outputs += [logit]
            alignments += [attention_weights]

        hidden_outputs, logit_outputs, alignments = \
            self.parse_decoder_outputs(hidden_outputs, logit_outputs, alignments)

        return hidden_outputs, logit_outputs, alignments

    def decode(self, decoder_input):
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            cell_input,
            (self.decoder_hidden, self.decoder_cell))

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
            self.attention_weights_cum.unsqueeze(1)),dim=1)

        self.attention_context, self.attention_weights = self.attention_layer(
            self.decoder_hidden,
            self.memory,
            self.processed_memory,
            attention_weights_cat,
            self.mask)

        self.attention_weights_cum += self.attention_weights
        hidden_and_context = torch.cat((self.decoder_hidden, self.attention_context), -1)
        hidden = self.project_to_hidden(hidden_and_context)
        logit = self.project_to_n_symbols(F.dropout(hidden, 0.5, self.training))
        return hidden, logit, self.attention_weights

    def parse_decoder_outputs(self, hidden, logit, alignments):
        alignments = torch.stack(alignments).transpose(0,1)
        logit = torch.stack(logit).transpose(0, 1).contiguous()
        hidden = torch.stack(hidden).transpose(0, 1).contiguous()
        return hidden, logit, alignments

class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = nn.Linear(attention_rnn_dim, attention_dim, bias=False)
        self.memory_layer = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(
            self,
            query,
            processed_memory,
            attention_weights_cat):
        """
        query: size=(B, H)
        processed_memory: size=(B, Lm, H)
        attention_weights_cat: size=(B, Lm, 2)


        """
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(
            self,
            attention_hidden_state,
            memory,
            processed_memory,
            attention_weights_cat, mask):
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)
        return attention_context, attention_weights

class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = nn.Conv1d(2, attention_n_filters,
                                       kernel_size=attention_kernel_size,
                                       padding=padding, bias=False, stride=1,
                                       dilation=1)
        self.location_dense = nn.Linear(attention_n_filters, attention_dim, bias=False)

    def forward(self, attention_weights_cat):
        """
        Args
        - attention_weights_cat: size=(B, 2, L)
        """
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


