# https://www.pytorch123.com/ThirdSection/DeployingSeq2SeqModelwithHybridFrontend/
# 混合前端的seq2seq模型部署:如何使seq2seq模型转换为PyTorch可用的前端混合Torch脚本
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from json import encoder
from unittest.util import _MAX_LENGTH
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os
import unicodedata
import numpy as np

import os
rootDir = './DATA/9_data'
os.makedirs(rootDir, exist_ok= True)
#这里有11种方法，供你用Python下载文件,https://zhuanlan.zhihu.com/p/587382385
import wget
url = "https://download.pytorch.org/models/tutorials/4000_checkpoint.tar"
filePath = rootDir + '/4000_checkpoint.tar'
if ( not os.path.isfile( filePath ) ):
    wget.download(url, filePath )

device = torch.device( "cpu" )
MAX_LENGTH = 10 # 最大句子长度
PAD_token  = 0  # 默认的词向量
SOS_token  = 1
EOS_token  = 2

# 4：数据处理
class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # 统计SOS, EOS, PAD


    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))
        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # 统计默认的令牌
        for word in keep_words:
            self.addWord(word)

# 小写并删除非字母字符
def normalizeString(s):
    s = s.lower()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# 使用字符串句子，返回单词索引的句子
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


# 5：定义编码器
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # 初始化GRU;input_size和hidden_size参数都设置为'hidden_size'
        # 因为我们输入的大小是一个有多个特征的词向量== hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # 将单词索引转换为向量
        embedded = self.embedding(input_seq)
        # 为RNN模块填充批次序列
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # 正向通过GRU
        outputs, hidden = self.gru(packed, hidden)
        # 打开填充
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # 将双向GRU的输出结果总和
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # 返回输出以及最终的隐藏状态
        return outputs, hidden


# 6：6.定义解码器的注意力模块
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # 根据给定的方法计算注意力权重（能量）
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # 转置max_length和batch_size维度
        attn_energies = attn_energies.t()

        # 返回softmax归一化概率分数（增加维度）
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


# 7：定义解码器
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = embedding
        self.attn = Attn(attn_model, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # 获取嵌入
        embedded = self.embedding(input_step)
        embedded = self.dropout(embedded)

        # GRU的输入是嵌入的单词索引
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # 计算注意力权重
        attn_weights = self.attn(rnn_output, encoder_outputs)

        # 将注意力权重应用于编码器输出
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        # 将RNN输出和上下文连接起来
        output = torch.cat((rnn_output[0], context[0]), 1)

        # 预测下一个单词
        output = self.out(output)

        # 返回输出和隐藏状态
        return output, hidden, attn_weights


# 8.1：定义评估，贪婪搜索解码器
class GreedySearchDecoder( torch.jit.ScriptModule ):
    def __init__( self, encoder, decoder, decoder_n_layers ):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._device = device
        self._SOS_token = SOS_token
        self._decoder_n_layers = decoder_n_layers

    __constants__ = ['_device', '_SOS_token', '_docoder_n_layers']

    @torch.jit.script_method
    def forward(self, input_seq : torch.Tensor, input_length : torch.Tensor, max_length : int ):
        # 通过编码器模型转发输入
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # 准备编码器的最终隐藏层作为解码器的第一个隐藏输入
        decoder_hidden = encoder_hidden[:self._decoder_n_layers]
        # 使用SOS_token初始化解码器输入
        decoder_input = torch.ones(1, 1, device=self._device, dtype=torch.long) * self._SOS_token
        # 初始化张量以将解码后的单词附加到
        all_tokens = torch.zeros([0], device=self._device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self._device)
        # 一次迭代地解码一个词tokens
        for _ in range(max_length):
            # 正向通过解码器
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # 获得最可能的单词标记及其softmax分数
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # 记录token和分数
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # 准备当前令牌作为下一个解码器输入（添加维度）
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # 返回收集到的词tokens和分数
        return all_tokens, all_scores


# 8.2：输入评估
# 接受一个字符串输入语句作为参数，对其进行规范化、计算并输出响应。
def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    # 格式化输入句子作为批处理
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # 创建长度张量
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # 转置批量的维度以匹配模型的期望
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # 使用适当的设备
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # 用earcher解码句子s
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words

# 评估来自用户输入的输入(stdin)
def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # 获取输入的句子
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # 规范化句子
            input_sentence = normalizeString(input_sentence)
            # 评估句子
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # 格式化和打印回复句
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")

# 规范化输入句子并调用evaluate()
def evaluateExample(sentence, encoder, decoder, searcher, voc):
    print("> " + sentence)
    # 规范化句子
    input_sentence = normalizeString(sentence)
    # 评估句子
    output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
    output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
    print('Bot:', ' '.join(output_words))


# 9：加载预训练参数
encoder = 0
decoder = 0
voc = Voc(0)
def Exp_seq2seq():
    save_dir = os.path.join("data", "save")
    corpus_name = "cornell movie-dialogs corpus"

    # 配置模型
    model_name = 'cb_model'
    attn_model = 'dot'
    #attn_model = 'general'
    #attn_model = 'concat'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    # 如果你加载的是自己的模型
    # 设置要加载的检查点
    checkpoint_iter = 4000
    # loadFilename = os.path.join(save_dir, model_name, corpus_name,
    #                             '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
    #                             '{}_checkpoint.tar'.format(checkpoint_iter))

    # 如果你加载的是托管模型
    loadFilename = rootDir + '/4000_checkpoint.tar'

    # 加载模型
    # 强制CPU设备选项（与本教程中的张量匹配）
    checkpoint = torch.load(loadFilename, map_location=torch.device('cuda:0'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc = Voc(corpus_name)
    voc.__dict__ = checkpoint['voc_dict']


    print('Building encoder and decoder ...')
    # 初始化词向量
    embedding = nn.Embedding(voc.num_words, hidden_size)
    embedding.load_state_dict(embedding_sd)
    # 初始化编码器和解码器模型
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    # 加载训练模型参数
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    # 使用适当的设备
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    # 将dropout层设置为eval模式
    encoder.eval()
    decoder.eval()
    print('Models built and ready to go!')

    ### 转换编码器模型
    # 创建人工输入
    test_seq = torch.LongTensor(MAX_LENGTH, 1).random_(0, voc.num_words).to(device)
    test_seq_length = torch.LongTensor([test_seq.size()[0]]).to(device)
    # 跟踪模型
    traced_encoder = torch.jit.trace(encoder, (test_seq, test_seq_length))

    ### 转换解码器模型
    # 创建并生成人工输入
    test_encoder_outputs, test_encoder_hidden = traced_encoder(test_seq, test_seq_length)
    test_decoder_hidden = test_encoder_hidden[:decoder.n_layers]
    test_decoder_input = torch.LongTensor(1, 1).random_(0, voc.num_words)
    # 跟踪模型
    traced_decoder = torch.jit.trace(decoder, (test_decoder_input, test_decoder_hidden, test_encoder_outputs))

    ### 初始化searcher模块
    scripted_searcher = GreedySearchDecoder(traced_encoder, traced_decoder, decoder.n_layers)



    # 评估例子
    sentences = ["hello", "what's up?", "who are you?", "where am I?", "where are you from?"]
    for s in sentences:
        evaluateExample(s, traced_encoder, traced_decoder, scripted_searcher, voc)

    # 评估你的输入
    #evaluateInput(traced_encoder, traced_decoder, scripted_searcher, voc)

if __name__=='__main__':
    Exp_seq2seq()

