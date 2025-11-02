import torch
from einops import rearrange
from torch import nn
import math
import torch.nn.functional as F
from transformers.models.cvt.convert_cvt_original_pytorch_checkpoint_to_pytorch import attention


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
        self.d = configs.transformer_d_model
        self.expansion = configs.transformer_forward_expansion
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
        self.n_layers = configs.transformer_encoder_n_layers
        self.n_frames = configs.n_frames
        self.device = configs.device
        self.d = configs.transformer_d_model
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


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, maxlen, device):
        super(PositionalEmbedding, self).__init__()
        self.encoding = torch.zeros(maxlen, d_model, device=device)
        self.encoding.requires_grad_(False)

        pos = torch.arange(0, maxlen, device=device).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, 2, device=device)

        self.encoding[:, 0::2] = torch.sin(pos / (1000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (1000 ** (_2i / d_model)))

    def forward(self, x):
        seq_len = x.shape[1]
        return self.encoding[:seq_len, :].unsqueeze(0).permute(0, 2, 1)


class PositionalEmbeddingLearnable(nn.Module):
    def __init__(self, maxlen, d_model, device):
        super(PositionalEmbeddingLearnable, self).__init__()

        # 初始化一个可训练参数
        self.encoding = nn.Parameter(torch.zeros(maxlen, d_model, device=device))

        # 正弦余弦初始化
        pos = torch.arange(0, maxlen, device=device).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, 2, device=device).float()

        pe = torch.zeros(maxlen, d_model, device=device)
        pe[:, 0::2] = torch.sin(pos / (1000 ** (_2i / d_model)))
        pe[:, 1::2] = torch.cos(pos / (1000 ** (_2i / d_model)))

        # 用 sin-cos 初始化，但允许训练时更新
        self.encoding.data.copy_(pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        return self.encoding[:seq_len, :].unsqueeze(0)


class ConvFFNBlock(nn.Module):
    def __init__(self, d, f, r, one=True):  # one is True: ConvFFN1, one is False: ConvFFN2
        super(ConvFFNBlock, self).__init__()
        groups_num = d if one else f
        self.pw_con1 = nn.Conv1d(
            in_channels=d * f,
            out_channels=r * d * f,
            kernel_size=1,
            groups=groups_num
        )
        self.pw_con2 = nn.Conv1d(
            in_channels=r * d * f,
            out_channels=d * f,
            kernel_size=1,
            groups=groups_num
        )

    def forward(self, x):
        # x: [B, f*f, N]
        x = self.pw_con1(x)
        x = F.gelu(x)
        x = self.pw_con2(x)
        return x
        # x = self.pw_con2(F.gelu(self.pw_con1(x)))


class ConvFFN(nn.Module):
    def __init__(self, d=6, f=6, kernel_size=31, r=4):
        super(ConvFFN, self).__init__()
        # 深度分离卷积负责捕获时域关系
        self.dw_conv = nn.Conv1d(
            in_channels=d * f,
            out_channels=d * f,
            kernel_size=kernel_size,
            groups=d * f,
            padding='same',
            # padding=1,
            # stride=4
        )

        self.bn = nn.BatchNorm1d(d * f)
        self.conv_ffn1 = ConvFFNBlock(d, f, r, True)
        self.conv_ffn2 = ConvFFNBlock(d, f, r, False)

    def forward(self, x_emb):
        b, l, d, f = x_emb.shape
        x = x_emb.view(b, d, f, l)
        x = rearrange(x, 'b d f l -> b (d f) l')  # [b, d, f, l] -> [b, d*f, l]
        x = self.dw_conv(x)  # [b, d*f, l] -> [b, d*f, l]
        x = self.bn(x)  # [b, d*f, l] -> [b, d*f, l]
        x = self.conv_ffn1(x)  # [B, d*f, N] -> [B, d*f, N]

        x = rearrange(x, 'b (d f) l -> b d f l', d=d)  # [b, d*f, l] -> [b, d, f, l]
        x = x.permute(0, 2, 1, 3)  # [b, d, f, l] -> [b, f, d, l]
        x = rearrange(x, 'b f d l -> b (f d) l')  # [b, f, d, l] -> [b, f*d, l]

        x = self.conv_ffn2(x)  # [B, f*d, N] -> [B, f*d, N]

        x = rearrange(x, 'b (f d) n -> b f d n', d=d)  # [B, f*d, N] -> [B, f, d, N]
        x = x.permute(0, 3, 2, 1)  # [B, f, d, N] -> [B, d, f, N]

        out = x + x_emb

        return out


class IMU_Embedding(nn.Module):
    def __init__(self, configs):
        super(IMU_Embedding, self).__init__()
        self.n_frames = configs.n_frames
        self.use_patch = configs.use_patch
        if self.use_patch:
            self.n_frames = configs.patch_len * ((self.n_frames - configs.patch_len) // configs.stride + 1)
        self.joints_emb = configs.joints_emb
        self.raw_dim = configs.raw_dim
        self.device = configs.device
        self.pe = configs.transformer_PE

        if self.joints_emb:
            self.joint_emb = nn.Embedding(6, 6)
        if self.pe == 'b':
            self.pos_emb = nn.Embedding(self.n_frames, self.raw_dim)
        elif self.pe == 't':
            self.pos_emb = PositionalEmbedding(self.n_frames, self.raw_dim, self.device)

    def forward(self, src):
        # 8,6,1500,6
        if self.use_patch:
            batch, n_device, patch_num, patch_length, dim = src.shape
            src = src.view(batch, n_device, -1, dim)
        batch, n_device, length, dim = src.shape
        if self.joints_emb:
            joint_idx = torch.arange(6, device=src.device)  # [5]
            je = self.joint_emb(joint_idx)  # [5, D]
            je = je.view(1, 6, 1, -1)  # [1, 5, 1, D]
            src = src + je  # [B, 5, T, D] + [1, 5, 1, D] -> broadcast to [B, 5, T, D]
        if self.pe == 't':
            pe = self.pos_emb(src)
            src = src + pe
        elif self.pe == 'b':
            pos_idx = torch.arange(self.n_frames, device=src.device)
            pe = self.pos_emb(pos_idx)
            pe = pe.view(1, self.n_frames, -1)
            src = src + pe
        return src



class IMU_Encoder(nn.Module):
    def __init__(self, configs):
        super(IMU_Encoder, self).__init__()
        self.raw_dim = configs.raw_dim
        self.d = configs.model_d_model
        self.conv_ffn = configs.conv_ffn
        self.caption_max_len = configs.caption_max_len
        self.n_frames = configs.n_frames
        self.use_patch = configs.use_patch
        if self.use_patch:
            self.n_frames = configs.patch_len * ((self.n_frames - configs.patch_len) // configs.stride + 1)
        self.imu_embedding = IMU_Embedding(configs)
        if self.conv_ffn:
            self.conv = ConvFFN()
        self.linear1 = nn.Linear(self.raw_dim, self.d)
        self.linear2 = nn.Linear(self.n_frames, self.caption_max_len)
        self.attn = Encoder(configs)

    def forward(self, src):
        # 8,6,1500,6
        src = self.imu_embedding(src)  # 8,6,1500,6
        src = src.view(-1, self.n_frames, 6, 6)
        if self.conv_ffn:
            src = self.conv(src)
        src = src.reshape(-1, self.n_frames, 36)
        src = self.linear1(src)
        enc_out = self.attn(src, src, src).permute(0, 2, 1).contiguous()
        enc_out = self.linear2(enc_out).permute(0, 2, 1).contiguous() # 8,45,512
        return enc_out


class Text_Encoder(nn.Module):
    def __init__(self, configs):
        super(Text_Encoder, self).__init__()
        self.device = configs.device
        self.d = configs.transformer_d_model
        self.vocab_size = configs.vocab_size
        self.caption_max_len = configs.caption_max_len
        self.word_embedding = nn.Embedding(556, self.d)
        self.position_embedding = nn.Embedding(self.caption_max_len, self.d)
        self.proj = nn.Linear(self.d, self.vocab_size)
        self.dropout = nn.Dropout(configs.dropout)
        self.attn = Encoder(configs)
        self.norm = nn.LayerNorm(self.d)

    def forward(self, x):
        mask = self.make_trg_mask(x)
        batch_size, seq_len = x.size()
        word_emb = self.word_embedding(x)
        positions = torch.arange(seq_len, device=self.device)
        pos_emb = self.position_embedding(positions)
        x = word_emb + pos_emb
        x = self.dropout(x)
        text_encode = self.attn(x, x, x, mask)
        text_encode = self.dropout(self.norm(text_encode + x))

        return text_encode

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

class LanguageGenerate(nn.Module):
    def __init__(self, configs):
        super(LanguageGenerate, self).__init__()

        self.fusion = EncoderBlock(configs)
        self.decoder = nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,556),
        )

    def forward(self, src, tgt):
        fusion = self.fusion(tgt,src,src)
        out = self.decoder(fusion)

        return out

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.d = configs.transformer_d_model
        self.raw_dim = configs.raw_dim
        self.caption_max_len = configs.caption_max_len
        self.device = configs.device
        self.use_patch = configs.use_patch
        self.transformer_patch = configs.transformer_patch
        self.n_frames = configs.n_frames
        self.joints_emb = configs.joints_emb
        self.batch = configs.batch_size
        self.pe = configs.transformer_PE
        self.conv_ffn = configs.conv_ffn
        self.itc_loss = configs.itc_loss
        self.device_index = configs.device_index
        if self.use_patch:
            if self.transformer_patch:
                self.n_frames = (self.n_frames - configs.patch_len) // configs.stride + 1
                self.raw_dim = configs.patch_len * self.raw_dim
            else:
                self.n_frames = configs.patch_len * ((self.n_frames - configs.patch_len) // configs.stride + 1)

        if self.conv_ffn:
            self.conv = ConvFFN()
        self.Linear1 = nn.Linear(self.raw_dim, self.d)
        self.Linear2 = nn.Linear(self.n_frames, self.caption_max_len)
        self.proj = nn.Linear(64, 556)

        self.imu_encoder = IMU_Encoder(configs)
        self.text_encoder = Text_Encoder(configs)
        self.language_generator = LanguageGenerate(configs)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src_size: patch_(8,8,25,300,6) wopatch_(8,6,1500,6)
        src_encode = self.imu_encoder(src)
        text_encode = self.text_encoder(tgt)
        out = self.language_generator(src_encode, text_encode)
        if self.itc_loss:
            t_f = text_encode[:, 0, :].squeeze(1)
            i_f = src_encode[:, 0, :].squeeze(1)
            loss = itc_loss(i_f, t_f)
            return out, loss
        return out, 0

    def device_mask(self, src):
        device_list = {
            1: (1, 0, 0, 0, 0, 0),
            2: (0, 1, 0, 0, 0, 0),
            3: (0, 0, 1, 0, 0, 0),
            4: (0, 0, 0, 1, 0, 0),
            5: (0, 0, 0, 0, 1, 0),
            6: (0, 0, 0, 0, 0, 1),
            7: (1, 0, 1, 0, 0, 0),
            8: (0, 0, 1, 1, 0, 0),
            9: (0, 0, 1, 0, 0, 1),
            10: (1, 0, 0, 0, 0, 1),
            11: (1, 0, 1, 0, 1, 0),
            12: (1, 0, 1, 0, 0, 1),
            13: (0, 0, 1, 0, 1, 1),
            14: (1, 0, 1, 0, 1, 1),
            15: (1, 1, 1, 1, 1, 1),
            16: (1, 1, 1, 1, 1, 0),
            17: (1, 1, 1, 1, 0, 1),
            18: (0, 1, 0, 1, 1, 0),
            19: (1, 0, 0, 1, 0, 1),
            20: (0, 1, 1, 0, 1, 1),
            21: (1, 1, 0, 0, 0, 0),
            22: (0, 1, 0, 0, 0, 1),
            23: (0, 0, 0, 1, 0, 1),
            24: (1, 1, 0, 1, 0, 0),
            25: (0, 0, 0, 0, 1, 1),
            26: (0, 1, 0, 1, 0, 0),
            27: (1, 0, 0, 1, 0, 0),
            28: (1, 0, 0, 0, 1, 0),
        }
        l = len(device_list)
        key = self.device_index
        batch, n_device, L, D = src.shape
        mask_table = torch.tensor(
            [device_list[i] for i in range(1, l + 1)],
            device=src.device, dtype=src.dtype
        )

        m = mask_table[key - 1]  # (6,)
        m = m.view(1, n_device, 1, 1)  # (1,6,1,1)
        return src * m  # 广播后 (batch,6,L,D)


def itc_loss(imu_feat, text_feat, temperature=0.07):
    """
    imu_feat: [B, D] imu特征
    text_feat: [B, D] 文本特征
    temperature: 控制对比强度的缩放因子
    """
    # 1. 特征归一化
    imu_feat = F.normalize(imu_feat, dim=-1)
    text_feat = F.normalize(text_feat, dim=-1)

    # 2. 相似度矩阵（双向计算）
    logits_per_image = imu_feat @ text_feat.T  # [B, B]
    logits_per_text = text_feat @ imu_feat.T  # [B, B]

    # 3. 除以温度系数
    logits_per_image /= temperature
    logits_per_text /= temperature

    # 4. 构造 ground truth：每个样本只匹配它自己
    labels = torch.arange(imu_feat.size(0), device=imu_feat.device)

    # 5. 计算交叉熵损失（双向）
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)

    # 6. 平均两个方向
    loss = (loss_i2t + loss_t2i) / 2
    return loss
