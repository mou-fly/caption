# import math
# import os
# import time

# import numpy as np
# import pandas as pd
# from numpy.core.shape_base import block
# from torch import nn, optim
# import torch
# import clip
# from PIL import Image
# from tqdm import tqdm


# class Attention(nn.Module):
#     def __init__(self, configs):
#         super(Attention, self).__init__()

#         self.n_h = configs.transformer_n_heads  # 8
#         self.d = configs.transformer_d_model  # 512
#         self.d_k = self.d // self.n_h  # 64
#         self.raw_dim = configs.raw_dim

#         self.w_q = nn.Linear(self.d, self.d)
#         self.w_k = nn.Linear(self.d, self.d)
#         self.w_v = nn.Linear(self.d, self.d)
#         self.softmax = nn.Softmax(dim=-1)
#         self.proj = nn.Linear(self.d, self.d)

#     def forward(self, q, k, v, mask=None):
#         batch, length, dim = q.shape
#         l_kv = k.shape[1]
#         q = self.w_q(q)
#         k = self.w_k(k)
#         v = self.w_v(v)
#         # q 1 30 512
#         # v 1 1  512

#         q = q.view(batch, length, self.n_h, self.d_k).permute(0, 2, 1, 3)
#         k = k.view(batch, l_kv, self.n_h, self.d_k).permute(0, 2, 1, 3)
#         v = v.view(batch, l_kv, self.n_h, self.d_k).permute(0, 2, 1, 3)  # 1 8 1 64

#         score = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)  # 1 8 30 30
#         if mask is not None:
#             score = score.masked_fill(mask == 0, -1e20)
#         score = self.softmax(score) @ v
#         score = score.permute(0, 2, 1, 3).contiguous().view(batch, length, -1)

#         return self.proj(score)

# class pretrainModel(nn.Module):
#     def __init__(self, args):
#         super(pretrainModel, self).__init__()
#         self.device = args.device
#         self.batch = args.batch_size if args.mode == 0 else 1
#         self.img_train_path = "C:\project\caption\data\processed_data_joint_30s_small\pic_train"
#         self.img_test_path  = "C:\project\caption\data\processed_data_joint_30s_small\pic_test"

#         self.train_label = pd.read_csv("C:\project\caption\data\processed_data_joint_30s_small\label_train.csv")[['caption']].values
#         self.test_label = pd.read_csv("C:\project\caption\data\processed_data_joint_30s_small\label_test.csv")[['caption']].values

#         self.img_mlp = nn.Sequential(
#             nn.Linear(in_features=512*5, out_features=1024),
#             nn.ReLU(),
#             nn.Linear(in_features=1024, out_features=512),
#         ).to(self.device)
#         self.text_mlp = nn.Sequential(
#             nn.Linear(in_features=512, out_features=512),
#             nn.ReLU(),
#             nn.Linear(in_features=512, out_features=512),
#         ).to(self.device)


#         self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
#         for param in self.model.parameters():
#             param.requires_grad = False

#     def forward(self, ids):
#         image_paths = []
#         for name in ids:
#             name = str(int(name))
#             seconde_path = os.path.join(self.img_train_path, name)
#             for i in range(5):
#                 image_paths.append(os.path.join(seconde_path, "device_{}.png".format(i)))
#         images = [self.preprocess(Image.open(p).convert("RGB")) for p in image_paths]  # ensure RGB
#         image_tensor = torch.stack(images).to(self.device)
#         image_features = self.model.encode_image(image_tensor)   # 40,512
#         image_features = image_features.float()
#         image_features = image_features.view(self.batch, -1)
#         image_features = self.img_mlp(image_features)

#         caption_list = []
#         for id in ids:
#             id = int(id)
#             caption_list.append(str(self.train_label[id]))

#         text_tokens = clip.tokenize(caption_list).to(self.device)  #8,512  # 8,77
#         text_features = self.model.encode_text(text_tokens).float()
#         text_features = self.text_mlp(text_features)

#         return image_features, text_features

# class EncoderBlock(nn.Module):
#     def __init__(self, configs):
#         super(EncoderBlock, self).__init__()
#         self.d = configs.vtar_d_model
#         self.expansion = configs.vtar_forward_expansion
#         self.dropout = configs.dropout


#         self.attn = Attention(configs)
#         self.norm1 = nn.LayerNorm(self.d)
#         self.norm2 = nn.LayerNorm(self.d)

#         self.ffn = nn.Sequential(
#             nn.Linear(self.d, self.d * self.expansion),
#             nn.GELU(),
#             nn.Linear(self.d * self.expansion, self.d),
#         )
#         self.dropout = nn.Dropout(self.dropout)

#     def forward(self, q, k, v, mask=None):
#         attention = self.attn(q, k, v, mask)
#         x = self.dropout(self.norm1(attention + q))
#         forward = self.ffn(x)
#         out = self.dropout(self.norm2(x + forward))
#         return out

# class Encoder(nn.Module):
#     def __init__(self, configs):
#         super(Encoder, self).__init__()
#         self.n_layers = configs.vtar_encoder_layers
#         self.n_frames = configs.n_frames
#         self.device = configs.device
#         self.d = configs.vtar_d_model
#         # self.position_embedding = nn.Embedding(self.n_frames, self.d)

#         self.layers = nn.ModuleList(
#             [
#                 EncoderBlock(configs)
#                 for _ in range(self.n_layers)
#             ]
#         )
#         self.dropout = nn.Dropout(configs.dropout)

#     def forward(self, q, k, v, mask=None):
#         # positions = torch.arange(0, length).expand(N, seq_length).to(self.device)

#         for layer in self.layers:
#             v = layer(q, k, v, mask=mask)

#         return v

# class Model(nn.Module):
#     def __init__(self, args):
#         super(Model, self).__init__()
#         self.pre = args.clip_pretrain
#         self.patience = args.patience
#         self.delta = args.delta
#         self.lr = args.lr
#         self.device = args.device
#         self.args = args

#         self.attn = Attention(args)
#         self.linear = nn.Linear(512,512)
#         self.word_embedding = nn.Embedding(556,512)
#         self.proj = nn.Linear(512,556)
#         self.dropout = nn.Dropout(args.dropout)
#         self.text_encoder = Encoder(args)

#     def forward(self, src, tgt, preModel):
#         img_features, text_features = preModel(src)
#         img_features = img_features.unsqueeze(1).expand(-1, 45, -1)
#         img_features = self.linear(img_features)  # (8,45,512)

#         tgt_mask = self.make_trg_mask(tgt)
#         word_emb = self.word_embedding(tgt)
#         tgt = self.dropout(word_emb)
#         text_encode = self.text_encoder(tgt, tgt, tgt, tgt_mask)
#         out = self.attn(text_encode, img_features, img_features)

#         return self.proj(out)

#     def make_trg_mask(self, trg):
#         N, trg_len = trg.shape
#         trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
#             N, 1, trg_len, trg_len
#         )
#         return trg_mask.to(self.device)

#     def pretrain(self, train_loader, val_loader, configs):
#         model = pretrainModel(configs)
#         best_loss = np.inf
#         cnt = 0
#         patience = configs.patience
#         delta = configs.delta
#         all_time = time.time()
#         criterion = self.CLIPContrastiveLoss()
#         model_optimizer = optim.Adam(
#             list(model.img_mlp.parameters()) + list(model.text_mlp.parameters()),
#             lr=configs.lr
#         )
#         for epoch in range(200):
#             iter_count = 0
#             train_loss = []
#             model.train()
#             epoch_time = time.time()
#             for i, (batch_x, batch_y, mask) in tqdm(enumerate(train_loader)):
#                 batch_x = torch.tensor(batch_x, dtype=torch.float32, device=configs.device)# 8,1
#                 batch_y = torch.tensor(batch_y, dtype=torch.long, device=configs.device)  # 8,45
#                 iter_count += 1
#                 model_optimizer.zero_grad()
#                 img_emb, txt_emb = model(batch_x)
#                 logits_img = img_emb @ txt_emb.t()
#                 logits_txt = txt_emb @ img_emb.t()
#                 labels = torch.arange(logits_img.size(0)).to(configs.device)
#                 loss = (criterion(logits_img, labels) + criterion(logits_txt, labels)) / 2
#                 loss.backward()
#                 model_optimizer.step()
#                 train_loss.append(loss.item())

#                 if (i + 1) % 100 == 0:
#                     print(
#                         "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
#                     iter_count = 0

#             valid_loss = self.val(configs, model, val_loader)

#             if valid_loss + delta < best_loss:
#                 best_loss = valid_loss
#                 torch.save(model.state_dict(), os.path.join("C:\project\caption\checkpoint\\clip_pretrain"))
#                 print("save better loss...")
#                 cnt = 0
#             else:
#                 cnt += 1
#                 if cnt >= patience:
#                     print("early stop")
#                     break
#             print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
#             print("Epoch: {0}, Train Loss: {1:.7f}".format(epoch + 1, np.mean(train_loss)))
#             print("Epoch: {0}, Valid Loss: {1:.7f}".format(epoch + 1, valid_loss))
#             print("patience : {} / {}".format(cnt, patience))

#     def val(self, configs, model, val_loader):
#         model.eval()
#         total_loss = 0
#         criterion = self.CLIPContrastiveLoss()
#         with torch.no_grad():
#             for batch_x, _, _ in val_loader:
#                 batch_x = batch_x.to(configs.device)
#                 img_emb, txt_emb = model(batch_x)
#                 logits_img = img_emb @ txt_emb.t()
#                 logits_txt = txt_emb @ img_emb.t()
#                 labels = torch.arange(logits_img.size(0)).to(configs.device)
#                 loss = (criterion(logits_img, labels) + criterion(logits_txt, labels)) / 2
#                 total_loss += loss.item()
#         return total_loss / len(val_loader)

#     def CLIPContrastiveLoss(self):
#         return nn.CrossEntropyLoss()

#     def load_pretrain(self):
#         preModel = pretrainModel(self.args)
#         model_path = "C:\project\caption\checkpoint\clip_pretrain"
#         preModel.load_state_dict(torch.load(model_path))
#         preModel.to(self.device)

#         return preModel

