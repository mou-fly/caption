import math

import h5py
import torch
from torch import nn
import torch.nn.functional as F
from torch.ao.quantization.fx import convert
from torch.onnx.symbolic_opset9 import contiguous

from utils.h5 import load_h5

class TextEncoder(nn.Module):
    def __init__(self, vocab_size=556, embedding_dim=512, hidden_dim=512):
        """
        vocab_size: 词表大小
        embedding_dim: 词向量维度（图中 Ws·st）
        hidden_dim: LSTM 隐藏单元数（图中 wt 的维度，注意 Bi-LSTM 会产出 2*hidden_dim）
        """
        super(TextEncoder, self).__init__()
        # 词嵌入矩阵 Ws
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 双向 LSTM，batch_first=True 方便输入 (B, L, D)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        # 注意力中的可学习向量 v，维度同 hidden_dim
        self.v = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x):
        """
        x: LongTensor of shape (batch_size, seq_len), 每个元素是词索引
        返回:
          - sentence_repr: Tensor of shape (batch_size, hidden_dim)，最终的句子表示 \bar{e}_w
          - alpha:          Tensor of shape (batch_size, seq_len)，每个位置的注意力权重 α_t
        """
        # 1) 词嵌入
        emb = self.embedding(x)                  # (B, L, D_embed)

        # 2) Bi-LSTM
        outputs, _ = self.lstm(emb)              # outputs: (B, L, 2*hidden_dim)

        # 3) 按公式 (→w_t + ←w_t) / 2
        h_dim = outputs.size(2) // 2
        h_f, h_b = outputs[:, :, :h_dim], outputs[:, :, h_dim:]
        h = (h_f + h_b) / 2                      # (B, L, hidden_dim)

        return h

class IMUEncoder(nn.Module):
    def __init__(self,
                 feat_dim=30,
                 final_hidden=256):
        """
        feat_dim: 输入的 IMU 特征维度 (e.g. 6：3 轴加速度 + 3 轴角速度)
        time_hidden: 时间维度 Bi-LSTM 输出隐状态大小（单向）
        final_hidden: 最终聚合后输出的维度 d
        """
        super().__init__()
        # 1) 时间编码器：双向 LSTM
        self.time_lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=final_hidden,
            bidirectional=True,
            batch_first=True,
        )
        self.linear1 = nn.Linear(1500, 45)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, database=None):
        B, S, T, feat_dim = x.shape  #(8,5,1500,6)
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, -1) #(8,1500,30)
        x = x.permute(0, 2, 1).contiguous()  # (8,30,1500)
        x = self.linear1(x).permute(0, 2, 1).contiguous() #(8,45,30)
        e_imu, _ = self.time_lstm(x)  #(8,45,512)
        if database is not None:
            q = e_imu.contiguous().view(B, -1) # (B,L)
            N = database.shape[0]
            k, v = database.view(N, -1), database.view(N, -1) # (N,L)
            out = []
            score = q @ k.transpose(0,1) / math.sqrt(512)  # (560,1,512)
            attn = self.softmax(score)
            e_imu = attn @ v
            out.append(e_imu)
            out = torch.stack(out, dim=1)
            return out.view(B, 45, -1)
        return e_imu

def get_data(device):
    h5_file = load_h5("/home/wangtiantian/dengfei/caption/data/rcg_database.h5")
    data_np = h5_file['data'][:]
    label_np = h5_file['label'][:]
    data_tensor = torch.from_numpy(data_np).float().to(device)
    data_tensor = data_tensor.view(-1, 45, 512)
    label_tensor = torch.from_numpy(label_np).float().to(device)
    label_tensor = label_tensor.view(-1, 45, 512)

    return data_tensor, label_tensor

class RetrieverModel(nn.Module):
    def __init__(self, args):
        super(RetrieverModel, self).__init__()
        self.k = 3
        self.imu_encoder = IMUEncoder()
    def forward(self, src, data, label):
        e_imu = self.imu_encoder(src, data)
        b, _, _ = e_imu.shape
        top_k_idx = []
        e_imu_ = e_imu
        e_imu = e_imu.view(b, -1)
        label = label.view(label.shape[0], -1)
        for imu in e_imu:
            cos_sim = F.cosine_similarity(imu, label, dim=-1)
            max_val, max_idx = torch.topk(cos_sim, k=self.k, largest=True)
            top_k_idx.append(max_idx)
        labels = []
        for i in range(b):
            choice = []
            for j in range(self.k):
                choice.append(label[top_k_idx[i][j]])
            choice = torch.stack(choice)
            labels.append(choice)
        labels = torch.stack(labels) # (8,5,512)
        labels = labels.view(b, -1, 512).contiguous() # (8,5*45,512)

        return e_imu_, labels

    def retrieval_loss(e_imu, e_text, margin=0.2):
        """
        e_imu, e_text: tensors of shape (B, D), 已经做了 L2 归一化
        margin: Δ

        返回:
          一个标量 loss
        """
        B, D = e_imu.size()
        # 1) 相似度矩阵 S[i,j] = sim(imu_i, text_j)
        #    因为是归一化向量，直接点积就是 cosine 相似度
        S = torch.matmul(e_imu, e_text.t())  # (B, B)

        # 2) 正例对的相似度：对角线
        pos = S.diag().view(B, 1)  # (B,1)

        # 3) 构造 video→text hinge loss
        #    loss_v2t[i,j] = max(0, Δ + S[i,j] - pos[i])
        loss_v2t = F.relu(margin + S - pos)  # (B, B)
        # 4) 构造 text→video hinge loss
        #    loss_t2v[j,i] = max(0, Δ + S[i,j] - pos[j])
        #    等价于对 S^T 做同样操作，再转回来
        loss_t2v = F.relu(margin + S.t() - pos)  # (B, B)

        # 5) 去掉对角（正例自己跟自己的对比不算负样本）
        diag_mask = torch.eye(B, device=S.device).bool()
        loss_v2t.masked_fill_(diag_mask, 0.)
        loss_t2v.masked_fill_(diag_mask, 0.)

        # 6) 平均所有正-负对
        #    总共有 2*B*(B-1) 个负样本对
        loss = (loss_v2t.sum() + loss_t2v.sum()) / (2 * B * (B - 1))
        return loss

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.data, self.label = get_data(args.device)
        self.device = args.device

        self.retriever = RetrieverModel(args)
        self.decoder = Attention(args)
        self.proj = nn.Linear(512,556)
        self.word_embedding = nn.Embedding(556, 512)
        self.text_encoder = Encoder(args)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, src, tgt):
        e_imu, labels = self.retriever(src, self.data, self.label)
        input = torch.cat((e_imu, labels), dim=1)
        tgt_mask = self.make_trg_mask(tgt)
        word_emb = self.word_embedding(tgt)
        tgt = self.dropout(word_emb)
        text_encode = self.text_encoder(tgt, tgt, tgt, tgt_mask)
        out = self.decoder(text_encode, input, input)

        return self.proj(out)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)


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