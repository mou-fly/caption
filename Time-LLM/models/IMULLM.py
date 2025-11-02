from math import sqrt

import torch
import torch.nn as nn
import transformers
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer

from layers.Embed import PatchEmbedding
from layers.StandardNorm import Normalize
from utils.LabelEncoder import decode_labels

transformers.logging.set_verbosity_error()


class MLP_enconde(nn.Module):
    def __init__(self, hidden_dim=32):
        super(MLP_enconde, self).__init__()
        # 输入维度为 6，隐藏层维度为 hidden_dim（可调），输出维度为 1
        self.linear1 = nn.Linear(6, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x 的形状是 [2, 4096, 6]
        # 首先通过 linear1，将维度从 6 映射到 hidden_dim
        x = self.linear1(x)  # 输出形状是 [2, 4096, hidden_dim]

        # 应用 ReLU 激活
        x = self.relu(x)  # 形状保持不变 [2, 4096, hidden_dim]

        # 通过 linear2 将维度从 hidden_dim 压缩到 1
        x = self.linear2(x)  # 输出形状是 [2, 4096, 1]

        return x


class MLP_deconde(nn.Module):
    def __init__(self, hidden_dim=32):
        super(MLP_deconde, self).__init__()
        # 输入维度为 1，隐藏层维度为 hidden_dim（可调），输出维度为 6
        self.linear1 = nn.Linear(1, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, 6)

    def forward(self, x):
        # x 的形状是 [2, 512, 1]
        # 首先通过 linear1，将维度从 1 映射到 hidden_dim
        x = self.linear1(x)  # 输出形状是 [2, 512, hidden_dim]

        # 应用 ReLU 激活
        x = self.relu(x)  # 形状保持不变 [2, 512, hidden_dim]

        # 通过 linear2 将维度从 hidden_dim 解码到 6
        x = self.linear2(x)  # 输出形状是 [2, 512, 6]

        return x

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)  # 从倒数第2个维度展平
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class LabelClassifyLayer(nn.Module):
    def __init__(self, label_dict, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.label_dict = label_dict

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)


class Model(nn.Module):

    # patch_len: 补丁长度，用于将输入序列分割成更小的片段。
    # stride: 步幅，用于在序列分割成补丁时的步长。
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.target = configs.target
        self.label_dict = configs.label_dict
        self.mlp_encode = MLP_enconde()
        self.mlp_decode = MLP_deconde()
        self.conv1d = nn.Conv1d(in_channels=6, out_channels=1, kernel_size=6)

        configs.llm_model = 'GPT2'

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained(r'C:\project\AutoTimes\gpt2_')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'C:\project\AutoTimes\gpt2_',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'C:\project\AutoTimes\gpt2_',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'C:\project\AutoTimes\gpt2_',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'C:\project\AutoTimes\gpt2_',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        # 设置tokenizer的填充标记（pad token）的。
        # 如果tokenizer已经有一个终止标记（end-of-sequence token, eos_token），就将其设为填充标记；否则，就添加一个新的填充标记。
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

            # prompt_domain布尔值，提示是否使用提示域
            if configs.prompt_domain:
                self.description = configs.content
            else:
                self.description = 'Inertial measurement unit (IMU) is a device that measures the three-axis attitude angle (or angular velocity) and acceleration of an object.'


        self.dropout = nn.Dropout(configs.dropout)

        # 将输入的序列分割成一系列固定长度的 patch，并使用 TokenEmbedding 进行嵌入编码
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        # 一个线性层，用于将词嵌入从词汇大小映射到 num_tokens。这可能是为了将模型的输出调整为特定的任务需求
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        # 重编程
        # configs.n_heads: 注意力头的数量，通常用于多头自注意力机制。
        # self.d_ff: 前馈网络的维度，在 Transformer 中通常是隐藏层的大小。
        # self.d_llm: 预训练语言模型的维度
        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        # patch数量
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        # 计算多头注意力机制中每个头所需的特征数量，通常是前馈网络的维度乘以补丁的数量
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def get_action_label(self, data_label):
        count_dict = {}
        for label in data_label:
            label_str = label
            if label_str in count_dict:
                count_dict[label_str] += 1
            else:
                count_dict[label_str] = 1



    def forward(self, x_enc, x_label, x_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_label, x_dec, self.label_dict) # [2,4096,1]
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_label, x_dec, label_dict):

        x_enc = self.normalize_layers(x_enc, 'norm')
        # x_enc_acc = x_enc[:,:,0:3]
        # x_enc_gyro = x_enc[:,:,3:6]
        # x_enc_acc = self.normalize_layers(x_enc[:,:,0:3],'norm')
        # x_enc_gyro = self.normalize_layers(x_enc[:,:,3:6],'norm')
        # x_enc = torch.cat((x_enc_acc, x_enc_gyro), dim=-1)
        # x_enc = self.mlp_encode(x_enc)
        # x_enc = x_enc.permute(0, 2, 1)
        # x_enc = self.conv1d(x_enc)
        # x_enc = x_enc.permute(0, 2, 1)

        # 原始的 x_enc 形状是 (B, T, N)，表示 B 个样本(批次)，每个样本有 T 个时间步，每个时间步有 N 个特征。
        B, T, N = x_enc.size()        # （2，4096，1）
        # permute 函数用于改变张量的维度顺序。具体来说，x_enc.permute(0, 2, 1) 将 x_enc 的维度从 (B, T, N) 变为 (B, N, T)
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)    # （2，1，512）->(2*1,512,1)

        x_shape = x_enc.shape
        b_scape = x_dec.shape[0]


        # # 调整label的shape（B,N,1)->(B*N,1)
        B_l, T_l, N_l = x_label.size()
        # x_label = x_label.permute(0, 2, 1).contiguous().reshape(B_l * N_l, T_l, 1)
        # 将其展平为二维，保留 batchsize 维度
        x_label = x_label.reshape(B_l, -1)
        # label_shape = x_label.size

        prompt = []
        for b in range(x_enc.shape[0]):  # 每一个独立的时间序列（共 B * N 个）进行的
            prompt_start = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"<Data description:This is the {str(self.target)} for a motion sequence;"
            )

            # 将label还原,通过计算label类别与数量生成sequence描述
            label = x_label[b]
            label = decode_labels(label_dict, label)
            prompt_sentence = self.count_categories(label)

            prompt_end = (
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
            )
            # 合并
            prompt_ = prompt_start + prompt_sentence + prompt_end
            prompt.append(prompt_)


        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()    # （B, T, N）  （2*1，512，1）->(2,1,512)->(2,512,1)

        # token化prompt
        # return_tensors这个参数告诉分词器返回 PyTorch 张量（pt 代表 PyTorch）。如果使用的是 TensorFlow，则可以将其设置为 "tf"。
        # padding和truncation分别表示是否（不足时）填充或（太长时）裁剪。
        # max_length设置了输入文本的最大长度为 2048 个标记（tokens）。如果输入的文本超过这个长度，分词器将会截断多余的部分。
        # .input_ids表示提取了分词器输出中的 input_ids，它是一个包含文本的整数 ID 的张量，代表了分词后的结果（也即保留文本所对应的ID张量）。
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids

        # embedding prompt ，形状为 (batch, prompt_token, dim)
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # [2,151,768]

        # self.word_embeddings 是词嵌入矩阵，形状为 (vocab_size, dim)。
        # permute(1, 0) 将其维度转换为 (dim, vocab_size)。
        # self.mapping_layer 应用一个映射层，将其映射到一个新的空间。
        # permute(1, 0) 将其维度恢复为 (vocab_size, dim)。
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()   # （B, N, T）  (2,4096,1)->(2,1,4096)

        # 将 x_enc 转换为 bfloat16 并通过补丁嵌入层，返回嵌入输出 enc_out 和变量数量 n_vars
        enc_out, n_vars = self.patch_embedding(x_enc)     # (2,257,32)  1

        # 重编程 enc_out是Q, source_embeddings（缩减后的词表）是K、V
        # enc_out [8, 65, 32]  source_embeddings [1000, 768]
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)  # (2,257,768)
        # enc_out [8, 65, 768]
        # 将提示嵌入和补丁嵌入连接在一起
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)    # (2,408,768)
        # 使用 LLM（如GPT）的模型进行推理，返回最后隐藏状态 dec_out。
        # dec = self.llm_model(inputs_embeds=llama_enc_out)
        # dec_out = dec.last_hidden_state
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state  # [2,408,768]

        # 截取隐藏状态的前 self.d_ff（fcn的维度个数） 个特征
        dec_out = dec_out[:, :, :self.d_ff]   # [2,408,128]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))    # (2,1,p,32)  torch.Size([2, 1, 408, 128])
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()    # (2,1,32,p)    torch.Size([2, 1, 128, 408])

        # 对解码输出的最后几个补丁进行输出投影
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])   # torch.Size([2, 1, 512])
        dec_out = dec_out.permute(0, 2, 1).contiguous()              # （2，1，96）->（2，96，1）  torch.Size([2, 512, 1])

        dec_out = self.mlp_decode(dec_out)

        # 进行反归一化
        dec_out_acc = self.normalize_layers(dec_out[:,:,0:3], 'denorm')
        dec_out_gyro = self.normalize_layers(dec_out[:,:,3:6], 'denorm')
        # dec_out = self.normalize_layers(dec_out, 'denorm')
        dec_out = torch.cat((dec_out_acc, dec_out_gyro), dim=-1)

        return dec_out

    def count_categories(self, data_label):
        """
        统计类别及其出现次数,并生成描述
        :param data_label
        :return: sequence描述
        """
        # 使用字典统计类别及其出现次数
        count_dict = {}
        for label in data_label:
            label_str = label
            if label_str in count_dict:
                count_dict[label_str] += 1
            else:
                count_dict[label_str] = 1

        parts = []
        for label, count in count_dict.items():
            parts.append(f" {count} time steps are '{label}' actions")

        # 生成句子
        if parts:
            # 将所有部分连接，并在最后一个部分前加上 "and then"
            sentence = "The first " + parts[0]
            for part in parts[1:]:
                sentence += ", and then " + part
            sentence += ";"

        return sentence


    def calcute_lags(self, x_enc):
        """
        通过计算序列的自相关性来确定滞后值（lags）。这是通过快速傅里叶变换（FFT）和逆快速傅里叶变换（iFFT）来实现的。
        也就是说这里计算的是patch之间前top_k个滞后值中自相关最强的时间点
        """
        # (B * N, T, 1)->(B*N, 1, T)
        # 对重排后的张量在时间维度（dim=-1 即最后一个维度）上进行快速傅里叶变换（Real FFT）。rfft 适用于实数输入，返回的是复数频谱
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        # 计算 q_fft 和 k_fft 的逐元素乘积，其中 torch.conj(k_fft) 是 k_fft 的共轭复数。
        # 这个操作实际上是在频域中计算自相关性。
        res = q_fft * torch.conj(k_fft)
        # 对结果 res 进行逆快速傅里叶变换（iFFT），将频域的自相关性结果转换回时域。
        corr = torch.fft.irfft(res, dim=-1)
        # 在特征维度（dim=1）上对自相关性结果取平均值。这样做是为了对每个特征的自相关性结果进行汇总，得到整体的自相关性概况。
        mean_value = torch.mean(corr, dim=1)
        # 在平均自相关性结果中找出前 top_k 个最大的值及其对应的索引（lags）。这些索引就是滞后值，表示在这些滞后步长上自相关性最强。
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags    # 返回前 top_k 个滞后值。
    # 自相关性（Autocorrelation），也称为序列相关性，是指一个时间序列与其自身在不同时间滞后（lag）下的相关性。自相关性是衡量时间序列数据在不同时间点上的相似程度的统计量。
    #  正自相关性：如果一个时间序列在某个滞后值下的自相关系数为正，说明在该滞后值下的时间点上的数据具有相似的趋势。
    #  负自相关性：如果自相关系数为负，说明在该滞后值下的时间点上的数据具有相反的趋势。
    #  无自相关性：如果自相关系数接近零，说明在该滞后值下的时间点上的数据没有明显的相关性。

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):  # 16 8 32 768
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)  # 32->1024
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)      # 768->1024
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)    # 768-1024
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)      # 1024->768
        self.n_heads = n_heads    # 8
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding