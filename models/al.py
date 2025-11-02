import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, args, vocab_size=556, raw_dim=30, hidden=500, dropout=0.5, max_len=45, n_frames=1500):
        super(Model, self).__init__()
        self.batch_size = args.batch_size
        self.raw_dim = raw_dim
        self.hidden = hidden
        self.max_len = max_len
        self.n_frames = n_frames
        self.vocab_size = vocab_size

        self.W_h = nn.Linear(hidden, hidden)  # 双向 LSTM 输出隐藏状态维度为 hidden_size*2
        self.U_a = nn.Linear(hidden, hidden)
        self.b_a = nn.Parameter(torch.zeros(hidden))
        self.w = nn.Parameter(torch.randn(hidden))

        self.drop = nn.Dropout(p=dropout)
        self.linear = nn.Linear(self.n_frames, self.max_len)
        self.linear1 = nn.Linear(self.raw_dim, self.hidden)
        self.linear2 = nn.Linear(self.hidden, self.vocab_size)

        self.lstm1 = nn.LSTM(hidden, hidden, batch_first=True, dropout=dropout, num_layers=1)
        self.lstm2 = nn.LSTM(2*hidden, hidden, batch_first=True, dropout=dropout)

        self.embedding = nn.Embedding(vocab_size, hidden)

    def forward(self, src, caption=None, training=True):
        b, d, l, f = src.shape
        src = src.permute(0, 2, 1, 3).contiguous().view(b, l, -1)
        src = src.permute(0, 2, 1)
        src = self.linear(src)
        src = src.permute(0, 2, 1).contiguous()
        src = self.drop(src)
        src = self.linear1(src)                   # src embed
        # src = src.view(-1, self.max_len, self.hidden)
        padding = torch.zeros([self.batch_size, self.max_len-1, self.hidden]).cuda()
        src = torch.cat((src, padding), 1)        # src input
        src_out, state_src = self.lstm1(src)   # (8,89,500)
        _ = src_out
        src_out = self.attention(src_out) + _

        if training:
            caption = self.embedding(caption[:, 0:self.max_len-1])    # (8,44,500)
            padding = torch.zeros([self.batch_size, self.max_len, self.hidden]).cuda()  # (8,45,500)
            caption = torch.cat((padding, caption), 1)        # caption padding    (8,89,500)
            caption = torch.cat((caption, src_out), 2)        # caption input      (8,89,1000)

            cap_out, state_cap = self.lstm2(caption)   # (8,89,500)
            # size of cap_out is [batch_size, 2*max_len-1, hidden]
            cap_out = cap_out[:, self.max_len:, :]     # (8,44,500)
            cap_out = cap_out.contiguous().view(-1, self.hidden)  # (352,500)
            cap_out = self.drop(cap_out)
            cap_out = self.linear2(cap_out)   # (352,556)
            return cap_out
            # cap_out size [batch_size*79, vocab_size]
        else:
            padding = torch.zeros([self.batch_size, self.max_len, self.hidden]).cuda()      # 8,45,500
            cap_input = torch.cat((padding, src_out[:, 0:self.max_len, :]), 2)  # 8,45,1000
            cap_out, state_cap = self.lstm2(cap_input)          # 8,45,500
            # padding input of the second layer of LSTM, 80 time steps

            bos_id = 1*torch.ones(self.batch_size, dtype=torch.long)  # eos
            bos_id = bos_id.cuda()
            cap_input = self.embedding(bos_id)  # 8,500
            cap_input = torch.cat((cap_input, src_out[:, self.max_len, :]), 1) # 8,1000
            cap_input = cap_input.view(self.batch_size, 1, 2*self.hidden)       # 8,1,1000

            cap_out, state_cap = self.lstm2(cap_input, state_cap)  # 8,1,500
            cap_out = cap_out.contiguous().view(-1, self.hidden)  # 8,500
            cap_out = self.drop(cap_out)
            cap_out = self.linear2(cap_out)   # 8,556
            cap_out = torch.argmax(cap_out, 1)
            # input ["<BOS>"] to let the generate start

            caption = []
            caption.append(cap_out)
            # put the generate word index in caption list, generate one word at one time step for each batch
            for i in range(self.max_len-2):
                cap_input = self.embedding(cap_out)
                cap_input = torch.cat((cap_input, src_out[:, self.max_len+1+i, :]), 1)
                cap_input = cap_input.view(self.batch_size, 1, 2 * self.hidden)

                cap_out, state_cap = self.lstm2(cap_input, state_cap)
                cap_out = cap_out.contiguous().view(-1, self.hidden)
                cap_out = self.drop(cap_out)
                cap_out = self.linear2(cap_out)
                cap_out = torch.argmax(cap_out, 1)
                # get the index of each word in vocabulary
                caption.append(cap_out)
            return caption
            # size of caption is [44, batch_size]

    def attention(self, lstm_output):
        e_t = torch.tanh(self.W_h(lstm_output) + self.U_a(lstm_output) + self.b_a)
        e_t = torch.matmul(e_t, self.w)
        alpha_t = F.softmax(e_t, dim=1)
        alpha_t = alpha_t.unsqueeze(-1)
        context_vector = alpha_t * lstm_output
        return context_vector