import torch
import torch.nn as nn
from models.vtar import Encoder, Attention
from modules.mamba import Mamba2Simple


class Projection(nn.Module):
    def __init__(self, args):
        super(Projection, self).__init__()
        self.conv = nn.Conv1d(36, 36, 31, padding='same')
        self.gn = nn.GroupNorm(6, 36)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 8,6,1500,6
        b, d, l, f = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().view(b, -1, l)
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        x = x.view(b, d, f, l).contiguous()
        return x


class Embeddings(nn.Module):
    def __init__(self, args):
        super(Embeddings, self).__init__()
        self.transformer = Encoder(args)
        self.linear = nn.Linear(36, 512)
        self.conv = nn.Conv1d(36, 512, 31, padding='same')
        self.convcat = nn.Conv1d(512*2, 512, 31, padding='same')

    def forward(self, x):
        b, d, f, l = x.shape
        x_t = x.permute(0, 3, 1, 2).contiguous().view(b, l, -1)
        x_t = self.linear(x_t)
        x_t = self.transformer(x_t, x_t, x_t).permute(0, 2, 1)
        x_c = x.view(b, -1, l)
        x_c = self.conv(x_c)  # 8,512,1500
        x_fusion = torch.cat((x_t, x_c), dim=1)
        embedding = self.convcat(x_fusion)
        return embedding.permute(0, 2, 1)


class IMU_Encoder(nn.Module):
    def __init__(self, args):
        super(IMU_Encoder, self).__init__()

        self.proj = Projection(args)
        self.embeddings = Embeddings(args)
        self.mamba = Mamba2Simple(512).to(args.device)

    def forward(self, x):
        x = self.proj(x)
        x = self.embeddings(x)
        x = self.mamba(x)
        return x


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.encoder_layers = args.vtar_encoder_layers
        self.device = args.device

        self.imu_encoder = IMU_Encoder(args)
        self.text_encoder = Encoder(args)

        self.proj = nn.Linear(512, 556)

        self.word_embedding = nn.Embedding(556, 512)
        self.dropout = nn.Dropout(args.dropout)

        self.decoder = Attention(args)

    def forward(self, src, tgt):
        b, d, l, f = src.shape

        src = self.imu_encoder(src) # 8,1500,512

        tgt_mask = self.make_trg_mask(tgt)
        word_emb = self.word_embedding(tgt)
        tgt = self.dropout(word_emb)
        text_encode = self.text_encoder(tgt, tgt, tgt, tgt_mask)
        dec = self.decoder(text_encode, src, src)
        out = self.proj(dec)
        return out

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)
