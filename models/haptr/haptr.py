import math

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = torch.add(x, self.pe[:x.size(-2), :])
        return x


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, dict_size=128, num_pos_feats=16):
        super().__init__()
        self.embed = nn.Embedding(dict_size, num_pos_feats)

    def forward(self, x):
        w = x.shape[-2]
        i = torch.arange(w, device=x.device)
        emb = self.embed(i)
        x = torch.add(x, emb)
        return x
    
class SignalEncoderConv(nn.Module):
    def __init__(self, num_patches, projection_dim, position=True, modalities=6, learnable=False):
        super().__init__()

        self.conv_1 = nn.Conv2d(1, 8, (3, 1))
        self.bn_1 = nn.BatchNorm2d(8)
        self.conv_2 = nn.Conv2d(8, 16, (3, 1))
        self.bn_2 = nn.BatchNorm2d(16)
        self.conv_3 = nn.Conv2d(16, 64, (3, 1))
        self.bn_3 = nn.BatchNorm2d(64)

        self.max = nn.AdaptiveMaxPool2d(4)

        if position:
            if learnable:
                self.position_embedding = LearnablePositionalEncoding(dict_size=num_patches,
                                                                      num_pos_feats=projection_dim)
            else:
                self.position_embedding = PositionalEncoding(projection_dim)

    def forward(self, inputs):
        x = F.relu(self.bn_1(self.conv_1(inputs.float())))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))

        x = self.max(x)

        x = x.flatten(-2, -1)
        x = x.unsqueeze(1)

        if self.position_embedding:
            x = self.position_embedding(x)

        return x


class SignalEncoderLinear(nn.Module):
    def __init__(self, num_patches, projection_dim, position=True, num_channels=6, learnable=False):
        super().__init__()

        self.num_patches = num_patches
        self.projection = nn.Sequential(
            nn.Linear(num_channels, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        if position:
            if learnable:
                self.position_embedding = LearnablePositionalEncoding(dict_size=num_patches,
                                                                      num_pos_feats=projection_dim)
            else:
                self.position_embedding = PositionalEncoding(projection_dim)

    def forward(self, inputs):
        x = self.projection(inputs)
        if self.position_embedding:
            x = self.position_embedding(x)

        return x


class HAPTR(nn.Module):
    def __init__(self, num_classes, projection_dim, sequence_length, nheads, num_encoder_layers, feed_forward,
                 dropout, dim_modalities, num_modalities):
        super().__init__()
        self.sequence_length = sequence_length
        self.dim_modalities = dim_modalities
        self.num_modalities = num_modalities

        # encodes a temporal position of values in timeseries
        self.signal_encoder = SignalEncoderLinear(sequence_length, projection_dim, num_channels=sum(dim_modalities),
                                                  learnable=False)

        # create a transformer layer that converts input timeseries into feature timeseries
        self.transformer = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=projection_dim,
                                                                                          nhead=nheads,
                                                                                          dropout=dropout,
                                                                                          dim_feedforward=feed_forward,
                                                                                          activation='gelu'),
                                                 num_layers=num_encoder_layers,
                                                 norm=nn.LayerNorm(projection_dim))

        # flatten each timestep with N channels into signle channel (each timestep separately)
        self.pooling = nn.AvgPool1d(projection_dim)

        # classify resulting timeseries
        self.mlp_head = nn.Sequential(
            nn.BatchNorm1d(sequence_length),
            nn.Dropout(dropout),
            nn.Linear(sequence_length, num_classes)
        )

    def forward(self, inputs):
        x = self.signal_encoder(inputs)
        transformer_input = x.squeeze(1).permute(1, 0, 2)
        x = self.transformer(transformer_input).permute(1, 0, 2)
        x = self.pooling(x).squeeze(-1)

        try:
            x = self.mlp_head(x)
        except ValueError:
            x = self.mlp_head(x)

        return x, {}  # no weights

    def warmup(self, device, num_reps=1, num_batch=1):
        for _ in range(num_reps):
            dummy_input = torch.randn((num_batch, self.sequence_length, sum(self.dim_modalities)),
                                      dtype=torch.float).to(device)
            self.forward(dummy_input)


class HAPTR_ModAtt(HAPTR):
    def __init__(self, num_classes, projection_dim, sequence_length, nheads, num_encoder_layers, feed_forward,
                 dropout, dim_modalities, num_modalities):
        super().__init__(num_classes, projection_dim, sequence_length, nheads, num_encoder_layers, feed_forward,
                         dropout, dim_modalities, num_modalities)

        self.mod_attn = ModalityAttention(num_modalities, dim_modalities, sequence_length, dropout)

        # encodes a temporal position of values in timeseries
        self.signal_encoder = SignalEncoderLinear(sequence_length, projection_dim,
                                                  num_channels=sum(dim_modalities) + num_modalities,
                                                  learnable=False)

    def forward(self, inputs):
        x_weighted, weights = self.mod_attn(inputs)
        x = torch.cat([torch.cat(inputs, -1), x_weighted.permute(1, 0, 2)], -1)
        x = self.signal_encoder(x)
        transformer_input = x.squeeze(1).permute(1, 0, 2)
        x = self.transformer(transformer_input).permute(1, 0, 2)
        x = self.pooling(x).squeeze(-1)
        x = self.mlp_head(x)
        return x, {"mod_weights": weights}

    def warmup(self, device, num_reps=1, num_batch=1):
        for _ in range(num_reps):
            dummy_input = list()

            for mod_dim in self.dim_modalities:
                shape = [num_batch, self.sequence_length, mod_dim]
                mod_input = torch.randn(shape, dtype=torch.float).to(device)
                dummy_input.append(mod_input)

            self.forward(dummy_input)


class ModalityAttention(nn.Module):
    def __init__(self, num_modalities, dim_modalities, sequence_length, dropout):
        super().__init__()
        assert len(dim_modalities) == num_modalities

        self.num_modalities = num_modalities
        self.dim_modalities = dim_modalities
        self.seq_length = sequence_length
        self.dropout = dropout

        # attention layer with one head
        self.self_attn = nn.MultiheadAttention(embed_dim=self.num_modalities,
                                               num_heads=1,
                                               dropout=self.dropout,
                                               kdim=self.seq_length,
                                               vdim=self.seq_length)

        # flatten modalities to obtain 1 weight per each
        self.flat_nn = nn.ModuleList([Conv1D(dim, 1, dropout) for dim in self.dim_modalities])

    def forward(self, inputs):
        mods = torch.stack([self.flat_nn[i](inputs[i].permute(0, 2, 1)) for i in range(self.num_modalities)])
        x, w = self.self_attn(query=mods.permute(2, 1, 0), key=mods, value=mods, need_weights=True)
        return x, w


class Conv1D(nn.Module):
    def __init__(self, dim_modality_in, dim_modality_out, dropout=0.1):
        super().__init__()
        self.flatten = nn.Sequential(
            nn.Conv1d(dim_modality_in, dim_modality_out, 1),
            nn.BatchNorm1d(dim_modality_out),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x, squeeze_output=True):
        '''
        Convolves each timestep with kernel separately.
        INPUT: BATCH x CHANNELS x LENGTH
        RETURNS: BATCH x LENGTH x CHANNELS
        '''
        x = self.flatten(x).permute(0, 2, 1)

        if squeeze_output:
            x = x.squeeze(-1)
        return x