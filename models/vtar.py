import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, maxlen, device):
        super(PositionalEmbedding, self).__init__()
        self.encoding = torch.zeros(maxlen, d_model, device=device)
        self.encoding.requires_grad_(False)

        pos = torch.arange(0, maxlen, device=device)
        pos = pos.float().unsqueeze(1)
        _2i = torch.arange(0, d_model, 2, device=device)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        seq_len = x.shape[1]
        return self.encoding[:seq_len, :]

class Attention(nn.Module):
    def __init__(self, configs):
        super(Attention, self).__init__()

        self.n_h = configs.transformer_n_heads  # 8
        self.d = configs.transformer_d_model  # 512
        self.d_k = self.d // self.n_h  # 64
        self.raw_dim = configs.raw_dim

        self.w_q = nn.Linear(self.d, self.d)
        self.w_k = nn.Linear(self.d, self.d)
        self.w_v = nn.Linear(self.d, self.d)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(self.d, self.d)

    def forward(self, q, k, v, mask=None):
        batch, length, dim = q.shape
        l_kv = k.shape[1]
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        # q 1 30 512
        # v 1 1  512

        q = q.view(batch, length, self.n_h, self.d_k).permute(0, 2, 1, 3)
        k = k.view(batch, l_kv, self.n_h, self.d_k).permute(0, 2, 1, 3)
        v = v.view(batch, l_kv, self.n_h, self.d_k).permute(0, 2, 1, 3)  # 1 8 1 64

        score = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)  # 1 8 30 30
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e20)
        score = self.softmax(score) @ v
        score = score.permute(0, 2, 1, 3).contiguous().view(batch, length, -1)

        return self.proj(score)

class EncoderBlock(nn.Module):
    def __init__(self, configs):
        super(EncoderBlock, self).__init__()
        self.d = configs.vtar_d_model
        self.expansion = configs.vtar_forward_expansion
        self.dropout = configs.dropout


        self.attn = Attention(configs)
        self.norm1 = nn.LayerNorm(self.d)
        self.norm2 = nn.LayerNorm(self.d)

        self.ffn = nn.Sequential(
            nn.Linear(self.d, self.d * self.expansion),
            nn.GELU(),
            nn.Linear(self.d * self.expansion, self.d),
        )
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, q, k, v, mask=None):
        attention = self.attn(q, k, v, mask)
        x = self.dropout(self.norm1(attention + q))
        forward = self.ffn(x)
        out = self.dropout(self.norm2(x + forward))
        return out

class Encoder(nn.Module):
    def __init__(self, configs):
        super(Encoder, self).__init__()
        self.n_layers = configs.vtar_encoder_layers
        self.n_frames = configs.n_frames
        self.device = configs.device
        self.d = configs.vtar_d_model
        # self.position_embedding = nn.Embedding(self.n_frames, self.d)

        self.layers = nn.ModuleList(
            [
                EncoderBlock(configs)
                for _ in range(self.n_layers)
            ]
        )
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, q, k, v, mask=None):
        # positions = torch.arange(0, length).expand(N, seq_length).to(self.device)

        for layer in self.layers:
            v = layer(q, k, v, mask=mask)

        return v

class retrieval(nn.Module):
    def __init__(self, args):
        super(retrieval, self).__init__()
        self.batch = args.batch_size

    def forward(self, imu, database_x, database_y):
        b = imu.shape[0]
        imu_ = imu.view(b, -1)
        # max_list = []
        res = []
        for data in imu_:
            cos_sim = F.cosine_similarity(data, database_x, dim=-1)
            max_val, max_idx = torch.max(cos_sim, dim=0)
            res.append(database_y[max_idx])
        return torch.stack(res)

class align():
    def __init__(self):
        super(align, self).__init__()

    def forward(self, imu, text, retrieval):
        return None

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.encoder_layers = args.vtar_encoder_layers
        self.device = args.device

        self.imu_encoder = Encoder(args)
        self.text_encoder = Encoder(args)
        self.fusion = Attention(args)
        self.retrieval = retrieval(args)

        self.linear1 = nn.Linear(1500,45)
        self.linear2 = nn.Linear(30,512)
        self.linear3 = nn.Linear(90,45)
        self.proj = nn.Linear(512,556)

        self.word_embedding = nn.Embedding(556, 512)
        self.dropout = nn.Dropout(args.dropout)


    def forward(self, src, tgt, database_x, database_y):
        b, d, l, f = src.shape
        add_info = self.retrieval(src, database_x, database_y).long()
        add_info = self.word_embedding(add_info)
        src = src.permute(0, 2, 1, 3).contiguous().view(b, l, -1)
        src = self.linear1(src.permute(0, 2, 1)).permute(0, 2, 1)
        src = self.linear2(src)
        src_encode = self.imu_encoder(src, src, src)
        tgt_mask = self.make_trg_mask(tgt)
        word_emb = self.word_embedding(tgt)
        tgt = self.dropout(word_emb)
        text_encode = self.text_encoder(tgt, tgt, tgt, tgt_mask)
        fusion_in = torch.cat((src_encode, add_info), dim=1).permute(0, 2, 1)
        fusion_in = self.linear3(fusion_in).permute(0, 2, 1).contiguous()

        fusion_out = self.fusion(text_encode, fusion_in, fusion_in)
        return self.proj(fusion_out)


    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)



