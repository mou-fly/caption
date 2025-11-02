import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vtar import Encoder

class Model(nn.Module):
    def __init__(self, args, input_size=512, hidden_size=512, vocab_size=556):
        super(Model, self).__init__()
        # imuEncoder
        self.encoder = Encoder(args)
        self.linear1 = nn.Linear(1500, 45)
        self.linear2 = nn.Linear(30, 512)
        # LSTM 层：处理输入序列，双向 LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # Attention 层的参数
        self.W_h = nn.Linear(hidden_size, hidden_size)  # 双向 LSTM 输出隐藏状态维度为 hidden_size*2
        self.U_a = nn.Linear(hidden_size, hidden_size)
        self.b_a = nn.Parameter(torch.zeros(hidden_size))
        self.w = nn.Parameter(torch.randn(hidden_size))

        # 输出层：将 LSTM 输出映射到字典大小 (vocab_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def attention(self, lstm_output):
        e_t = torch.tanh(self.W_h(lstm_output) + self.U_a(lstm_output) + self.b_a)
        e_t = torch.matmul(e_t, self.w)
        alpha_t = F.softmax(e_t, dim=1)
        alpha_t = alpha_t.unsqueeze(-1)
        context_vector = alpha_t * lstm_output
        return context_vector

    def forward(self, x):
        b, d, l, f = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, l, -1)
        x = self.linear1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.linear2(x)
        src_encode = self.encoder(x, x, x)

        lstm_out, (h_n, c_n) = self.lstm(src_encode)
        context_vector = self.attention(lstm_out)
        output = self.fc(context_vector)
        return output

