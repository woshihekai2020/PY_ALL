#https://www.pytorch123.com/FifthSection/CharRNNClassification/
#使用字符级RNN进行名字分类
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string               #1
import torch                #1.1
import torch.nn as nn       #2
import random               #3
import time                 #3.3
import math
import matplotlib           #3.4
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
rootDir = './DATA/23_data/'
os.makedirs(rootDir, exist_ok= True)    # check dir exist or not?

############################################################################################################## 1:准备数据
def findFiles(path): return glob.glob(path)
print(findFiles(rootDir + '/data/*.txt'))

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
def unicodeToAscii(s):
    # 将Unicode字符串转换为纯ASCII, 感谢https://stackoverflow.com/a/518232/2809427
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters)

print(unicodeToAscii('Ślusàrski'))              # 只保留字母和空格
category_lines = {}                             # 构建category_lines字典，每种语言的名字列表
all_categories = []

def readLines(filename):
    # 读取文件并分成几行
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles(rootDir + '/data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines
n_categories = len(all_categories)
print(category_lines['Italian'][:5])

# 1.1: 单词转变为张量
def letterToIndex(letter):
    # 从all_letters中查找字母索引，例如 "a" = 0
    return all_letters.find(letter)
def letterToTensor(letter):
    # 仅用于演示，将字母转换为<1 x n_letters> 张量
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor
def lineToTensor(line):
    # 将一行转换为<line_length x 1 x n_letters>，或一个0ne-hot字母向量的数组
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print(letterToTensor('J'))
print(lineToTensor('Jones').size())

########################################################################################################### 2:构建神经网络
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

input = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)
print(output)

################################################################################################################## 3:训练
# 3.1:训练前的准备
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i
print(categoryFromOutput(output))

                                            # 需要一种快速获取训练示例（得到一个名字及其所属的语言类别）的方法
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]
def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor
for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)

# 3.2:训练神经网络
criterion = nn.NLLLoss()
learning_rate = 0.005                       # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # 将参数的梯度添加到其值中，乘以学习速率
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

# 3.3:打印其输出结果并跟踪其损失画图,并求平均损失。
n_iters = 100000
print_every = 5000
plot_every = 1000

current_loss = 0                            # 跟踪绘图的损失
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()
for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    if iter % print_every == 0:             # 打印迭代的编号，损失，名字和猜测
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d  %d%% (%s) %.4f  %s / %s  %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    if iter % plot_every == 0:              # 将当前损失平均值添加到损失列表中
        all_losses.append(current_loss / plot_every)
        current_loss = 0

# 3.4:绘画出结果
plt.figure()
plt.plot(all_losses)

############################################################################################################## 4:评价结果
confusion = torch.zeros(n_categories, n_categories)             # 在混淆矩阵中跟踪正确的猜测
n_confusion = 10000

# 只需返回给定一行的输出
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# 查看一堆正确猜到的例子和记录
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# 通过将每一行除以其总和来归一化
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

fig = plt.figure()                                              # 设置绘图
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

ax.set_xticklabels([''] + all_categories, rotation=90)          # 设置轴
ax.set_yticklabels([''] + all_categories)

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))           # 每个刻度线强制标签
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()                                                      # sphinx_gallery_thumbnail_number = 2

########################################################################################################### 5:处理用户输入
def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        topv, topi = output.topk(n_predictions, 1, True)        # 获得前N个类别
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Satoshi')

