import torch
import torch.nn as nn
from models.vtar import Attention, Encoder

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.device = args.device

        self.text_encoder = Encoder(args)
        self.imu_encoder = Encoder(args)
        self.align1 = Attention(args)
        self.align2 = Attention(args)
        self.align3 = Attention(args)
        self.dropout = nn.Dropout(args.dropout)
        self.word_embedding = nn.Embedding(556, 512)

        self.linear = nn.Linear(36,512)

        self.proj = nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512, 556)
        )

    def forward(self, src, tgt):
        # 8,6,1500,6
        b, d, l, f = src.shape
        src = src.permute(0, 2, 1, 3).contiguous().view(b, l, -1)
        src = self.linear(src)
        src = self.imu_encoder(src, src, src)
        tgt_mask = self.make_trg_mask(tgt)
        word_emb = self.word_embedding(tgt)
        tgt = self.dropout(word_emb)
        text_encode = self.text_encoder(tgt, tgt, tgt, tgt_mask)

        fusion1 = self.align1(tgt, src, src)
        fusion1 = self.dropout(fusion1)
        # fusion2 = self.align2(fusion1, src, src)
        # fusion2 = self.dropout(fusion2)
        # fusion3 = self.align3(fusion2, src, src)

        return self.proj(fusion1)


    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

