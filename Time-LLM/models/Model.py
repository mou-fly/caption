import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer, FullAttention, FreqMLP
from layers.Embed import DataEmbedding
from layers.Autoformer_EncDec import series_decomp, FourierAttention, TA, FA, series_decomp_multi, HybridAttention
from layers.decomposition import Decomposition
from utils.RevIN import RevIN
import torch.nn.functional as F




class CrossAttentionGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, trend, season):
        combined = torch.cat([trend, season], dim=-1)
        gate = self.gate(combined)
        return gate * trend + (1 - gate) * season


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=2048,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x_o, attn1 = self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )
        x = x + self.dropout(x_o)
        x = self.norm1(x)

        x_o, attn2 = self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )
        x = x + self.dropout(x_o)

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))


        return self.norm3(x + y), (attn2)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x, attns


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # self.pred_len = configs.pred_len
        # self.label_len = configs.label_len
        # self.output_attention = configs.output_attention
        #
        # # Embedding
        # self.embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
        #                                configs.dropout)
        #
        # kernel_size = configs.moving_avg
        # kernel_size = [15, 25, 35]
        # self.decomp = series_decomp_multi(kernel_size)
        #
        # self.Decomposition_model = Decomposition(input_length=configs.seq_len,
        #                                          pred_length=configs.pred_len,
        #                                          wavelet_name=configs.wavelet_name,
        #                                          level=configs.level,
        #                                          batch_size=configs.batch_size,
        #                                          channel=configs.enc_in,
        #                                          d_model=configs.d_model,
        #                                          tfactor=configs.tfactor,
        #                                          dfactor=configs.dfactor,
        #                                          device=configs.gpu,
        #                                          no_decomposition=configs.no_decomposition,
        #                                          use_amp=configs.use_amp)
        #
        # self.input_w_dim = self.Decomposition_model.input_w_dim  # list of the length of the input coefficient series
        # self.pred_w_dim = self.Decomposition_model.pred_w_dim  # list of the length of the predicted coefficient series
        # self.embedding_layer = nn.Linear(configs.enc_in, configs.d_model)
        # self.TA = TA(
        #     [
        #         EncoderLayer(
        #             AttentionLayer(
        #                 FullAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                                               output_attention=configs.output_attention),
        #                 configs.d_model, configs.n_heads, FourierAttention=0,TemporalAttention=1),
        #             configs.d_model,
        #             configs.d_ff,
        #             dropout=configs.dropout,
        #             activation=configs.activation
        #         ) for l in range(configs.e_layers)
        #     ],
        #     norm_layer=torch.nn.LayerNorm(configs.d_model)
        # )
        #
        # self.FA = FA(
        #     [
        #         EncoderLayer(
        #             AttentionLayer(
        #                 FourierAttention(T=1, activation='softmax', output_attention=False),
        #                 configs.d_model,
        #                 configs.n_heads, FourierAttention=1,TemporalAttention=0),
        #             configs.d_model,
        #             configs.d_ff,
        #             dropout=configs.dropout,
        #             activation=configs.activation
        #         ) for l in range(configs.e_layers)
        #     ],
        #     norm_layer=torch.nn.LayerNorm(configs.d_model)
        # )
        #
        # self.seasonal_decoder = Decoder(
        #     [
        #         DecoderLayer(
        #             AttentionLayer(
        #                         FourierAttention(T=1, activation='softmax', output_attention=False),
        #                      configs.d_model,
        #                 configs.n_heads, FourierAttention=1,TemporalAttention=0),
        #             AttentionLayer(
        #                         FourierAttention(T=1, activation='softmax', output_attention=False),
        #                         configs.d_model,
        #                 configs.n_heads, FourierAttention=1,TemporalAttention=0),
        #             configs.d_model,
        #             dropout=configs.dropout,
        #             activation=configs.activation,
        #         )
        #         for l in range(1)
        #     ],
        #     norm_layer=torch.nn.LayerNorm(configs.d_model),
        # )
        #
        # self.lstm1 = nn.LSTM(
        #     input_size=configs.d_model,  # configs.enc_in
        #     hidden_size=configs.d_ff // 2,  # 双向LSTM会使输出维度翻倍
        #     dropout=configs.dropout,
        #     bidirectional=True,  # 双向LSTM
        #     batch_first=True
        # )
        # self.lstm2 = nn.LSTM(
        #     input_size=configs.d_ff,  # configs.enc_in
        #     hidden_size=configs.d_model // 2,  # 双向LSTM会使输出维度翻倍
        #     dropout=configs.dropout,
        #     bidirectional=True,  # 双向LSTM
        #     batch_first=True
        # )
        #
        # self.temporal_branches = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv1d(configs.d_model, configs.d_model, kernel_size=3, dilation=2 ** i, padding=2 ** i),
        #         nn.GELU(),
        #         nn.BatchNorm1d(configs.d_model)
        #     ) for i in range(3)
        # ])
        #
        # self.Fmlp = FreqMLP(configs.d_model)
        #
        # self.revin1 = RevIN(configs.enc_in, eps=1e-5, affine=True, subtract_last=False)
        # self.revin2 = RevIN(configs.enc_in, eps=1e-5, affine=True, subtract_last=False)
        # self.fmlp = FreqMLP(configs.d_model)
        # self.norm_trend = nn.LayerNorm(configs.d_model)  # d_model=512
        # self.norm_season = nn.LayerNorm(configs.d_model)
        # self.relu = nn.ReLU()
        #
        # self.gate = CrossAttentionGate(configs.d_model)
        # self.dropout = nn.Dropout(configs.dropout)
        # self.projection = nn.Linear(configs.d_model, configs.enc_in, bias=True)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, enc_self_mask=None, dec_self_mask=None,
                dec_enc_mask=None):
        # x = self.revin1(x_enc, 'norm')
        # x_season, x_trend = self.decomp(x)
        #
        #
        # x0 = F.pad(x_season[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        #
        #
        # x_trend = self.embedding(x_trend, x_mark_enc)
        # x1 = self.lstm1(x_trend)[0]
        # x1 = self.lstm2(x1)[0]
        # x11 = x_trend + x1
        # x1, _ = self.TA(x11)
        # trend = self.norm_trend(x11+x1)
        #
        # x_season = self.embedding(x_season, x_mark_enc)
        # x2 = self.fmlp(x_season)
        # x21 = self.relu(x2)
        # x22 = self.fmlp(x21)
        # x_season = x_season + x22
        # season, _ = self.FA(x_season)
        #
        # season = self.norm_season(season+x_season)
        #
        #
        #
        # out = self.gate(trend, season)
        #
        # dec_out = self.embedding(x0,x_mark_dec)
        # season_out, attn_d = self.seasonal_decoder(dec_out, out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        #
        #
        #
        # season_out= self.projection(self.dropout(season_out[:, -self.pred_len:, :]))

        #seasonal_ratio = season.abs().mean(dim=1) / season_out.abs().mean(dim=1)
        #seasonal_ratio = seasonal_ratio.unsqueeze(1).expand(-1, self.pred_len, -1)

        #y = trend + season_out
        # y = self.revin1(season_out, 'denorm')
        # y = y[:, -self.pred_len:, -1:]
        # attn1 = self.spa_att(x)
        # attn2 = self.eca_att(x)
        # spa_out = attn1 * x
        # eca_out = attn2 * x
        # x = spa_out + eca_out

        # print(enc_out.shape) torch.Size([32, 96, 512])

        return x_enc
