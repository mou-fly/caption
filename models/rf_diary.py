import torch
import torch.nn as nn

from models.vtar import Encoder, Attention

class SkeletonGenerator(nn.Module):
    def __init__(self, args):
        super(SkeletonGenerator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1500,300),
            nn.ReLU(),
            nn.Linear(300,45)
        )
        self.linear = nn.Linear(300, 45)
        self.lstm = nn.LSTM(
            input_size=30,
            hidden_size=256,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, x):  # (8,5,1500,6)
        b, d, l, f = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, l, -1).contiguous().permute(0, 2, 1)  #(8,30,1500)
        x = self.mlp(x).permute(0, 2, 1).contiguous()
        pose, _ = self.lstm(x)
        return pose

class SkeletonEncoder(nn.Module):
    def __init__(self, args):
        super(SkeletonEncoder, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.mlp = nn.Sequential(
            nn.Linear(32,128),
            nn.ReLU(),
            nn.Linear(128,512),
        )
    def forward(self, pose):
        pose = pose.permute(0, 2, 1).contiguous()
        pose_enc = self.relu(self.conv1(pose))
        pose_enc = self.relu(self.conv2(pose_enc))
        pose_enc = self.relu(self.conv3(pose_enc))
        pose_enc = pose_enc.permute(0, 2, 1).contiguous()
        pose_enc = self.mlp(pose_enc)
        pose_out = pose_enc

        return pose_out

class LanguageGenerator(nn.Module):
    def __init__(self, args):
        super(LanguageGenerator, self).__init__()
        self.device = args.device

        self.text_encoder = Encoder(args)
        self.word_embedding = nn.Embedding(556,512)
        self.dropout = nn.Dropout(args.dropout)
        self.attn = Attention(args)
        self.proj = nn.Linear(512,556)

    def forward(self, x, imu):
        word_emb = self.word_embedding(x)
        mask = self.make_trg_mask(x)
        caption = self.dropout(word_emb)
        cap_enc = self.text_encoder(caption, caption, caption, mask)
        dec = self.attn(cap_enc, imu, imu)
        return self.proj(dec)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.sg = SkeletonGenerator(args)
        self.lg = LanguageGenerator(args)
        self.se = SkeletonEncoder(args)

    def forward(self, src, tgt):
        pose = self.sg(src)
        # pose_enc = self.se(pose)
        pose_enc = pose
        out = self.lg(tgt, pose_enc)
        return out, None
