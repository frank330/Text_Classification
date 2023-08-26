# -*- coding: utf-8 -*-
from jieba import lcut
from torchtext.vocab import vocab
from collections import OrderedDict, Counter
from torchtext.transforms import VocabTransform
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from sklearn.preprocessing import LabelEncoder
import scipy.io as io
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from utils import metrics, safeCreateDir
import time
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import seaborn as sns
import torch
from torch.nn import functional as F
import math
from sklearn.metrics import confusion_matrix  # 从sklearn工具包导入评价指标：混淆矩阵和f1值

# 数据处理
# """判断一个unicode是否是汉字"""
def is_chinese(uchar):
    if (uchar >= '\u4e00' and uchar <= '\u9fa5') :
        return True
    else:
        return False

# 是中文就留下 不是就跳过
def reserve_chinese(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str += i
    return content_str

# 读取去停用词库
def getStopWords():
    file = open('./dataset/stopwords.txt', 'r',encoding='utf8')
    words = [i.strip() for i in file.readlines()]
    file.close()
    return words

# 数据清洗、分词、去停用词
def dataParse(text, stop_words):
    label_map = {'news_story': 0, 'news_culture': 1, 'news_entertainment': 2,
               'news_sports': 3, 'news_finance': 4, 'news_house': 5, 'news_car': 6,
               'news_edu': 7, 'news_tech': 8, 'news_military': 9, 'news_travel': 10,
               'news_world': 11, 'stock': 12, 'news_agriculture': 13, 'news_game': 14}
    _, _, label,content, _ = text.split('_!_')
    label = label_map[label]
    # 去掉非中文词
    content = reserve_chinese(content)
    # 结巴分词
    words = lcut(content)
    # 去停用词
    words = [i for i in words if not i in stop_words]
    return words, int(label)

def getFormatData():
    file = open('./dataset/data/toutiao_cat_data.txt', 'r',encoding='utf8')
    texts = file.readlines()
    file.close()
    stop_words = getStopWords()
    all_words = []
    all_labels = []
    for text in texts:
        content, label = dataParse(text, stop_words)
        if len(content) <= 0:
            continue
        all_words.append(content)
        all_labels.append(label)

    # 自制词表Vocab
    # 将所有词都汇总到一个列表中
    ws = sum(all_words, [])
    # 统计词频
    set_ws = Counter(ws)
    # 按照词频排序 sorted函数是默认升序排序，当需要降序排序时，需要使用reverse = Ture
    # 以词的形式进行索引
    keys = sorted(set_ws, key=lambda x: set_ws[x], reverse=True)
    # 将词和编号对应起来 制作成字典
    dict_words = dict(zip(keys, list(range(1, len(set_ws) + 1))))
    ordered_dict = OrderedDict(dict_words)
    # # 基于有序字典创建词典 添加特殊符号
    my_vocab = vocab(ordered_dict, specials=['<UNK>', '<SEP>'])

    # 将输入的词元映射成它们在词表中的索引
    vocab_transform = VocabTransform(my_vocab)
    vector = vocab_transform(all_words)

    # 转成tensor
    vector = [torch.tensor(i) for i in vector]
    # 对tensor做padding 保证网络定长输入
    pad_seq = pad_sequence(vector, batch_first=True)
    labelencoder = LabelEncoder()
    labels = labelencoder.fit_transform(all_labels)
    data = pad_seq.numpy()
    data = {'X': data,
            'label': labels,
            'num_words': len(my_vocab)}
    io.savemat('./dataset/data/data.mat', data)

# 数据集加载
class Data(Dataset):
    def __init__(self, mode='train'):
        data = io.loadmat('./dataset/data/data.mat')
        self.X = data['X']
        self.y = data['label']
        self.num_words = data['num_words'].item()
        train_X, val_X, train_y, val_y, = train_test_split(self.X, self.y.squeeze(),test_size=0.3, random_state=1)
        val_X, test_X, val_y, test_y, = train_test_split(val_X, val_y, test_size=0.5, random_state=1)
        if mode == 'train':
            self.X = train_X
            self.y = train_y

        elif mode == 'val':
            self.X = val_X
            self.y = val_y

        elif mode == 'test':
            self.X = test_X
            self.y = test_y

    def __getitem__(self, item):
        return self.X[item], self.y[item]
    def __len__(self):
        return self.X.shape[0]
class getDataLoader():
    def __init__(self,batch_size):
        train_data = Data('train')
        val_data = Data('val')
        test_data = Data('test')
        self.traindl = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        self.valdl = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)
        self.testdl = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
        self.num_words = train_data.num_words

# 定义网络结构
class textCNN(nn.Module):
    def __init__(self, param):
        super(textCNN, self).__init__()
        ci = 1  # input chanel size
        kernel_num = param['kernel_num'] # output chanel size
        kernel_size = param['kernel_size']
        vocab_size = param['vocab_size']
        embed_dim = param['embed_dim']
        dropout = param['dropout']
        class_num = param['class_num']
        self.param = param
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.conv11 = nn.Conv2d(ci, kernel_num, (kernel_size[0], embed_dim))
        self.conv12 = nn.Conv2d(ci, kernel_num, (kernel_size[1], embed_dim))
        self.conv13 = nn.Conv2d(ci, kernel_num, (kernel_size[2], embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_size) * kernel_num, class_num)

    def init_embed(self, embed_matrix):
        self.embed.weight = nn.Parameter(torch.Tensor(embed_matrix))

    @staticmethod
    def conv_and_pool(x, conv):
        # x: (batch, 1, sentence_length,  )
        x = conv(x)
        # x: (batch, kernel_num, H_out, 1)
        x = F.relu(x.squeeze(3))
        # x: (batch, kernel_num, H_out)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        #  (batch, kernel_num)
        return x

    def forward(self, x):
        # x: (batch, sentence_length)
        x = self.embed(x)
        # x: (batch, sentence_length, embed_dim)
        # TODO init embed matrix with pre-trained
        x = x.unsqueeze(1)
        # x: (batch, 1, sentence_length, embed_dim)
        x1 = self.conv_and_pool(x, self.conv11)  # (batch, kernel_num)
        x2 = self.conv_and_pool(x, self.conv12)  # (batch, kernel_num)
        x3 = self.conv_and_pool(x, self.conv13)  # (batch, kernel_num)
        x = torch.cat((x1, x2, x3), 1)  # (batch, 3 * kernel_num)
        x = self.dropout(x)
        logit = F.log_softmax(self.fc1(x), dim=1)
        return logit

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def plot_acc(train_acc):
        sns.set(style='darkgrid')
        plt.figure(figsize=(10, 7))
        x = list(range(len(train_acc)))
        plt.plot(x, train_acc, alpha=0.9, linewidth=2, label='train acc')
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.legend(loc='best')
        plt.savefig('results/acc.png', dpi=400)

def plot_loss(train_loss):
        sns.set(style='darkgrid')
        plt.figure(figsize=(10, 7))
        x = list(range(len(train_loss)))
        plt.plot(x, train_loss, alpha=0.9, linewidth=2, label='train loss')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.legend(loc='best')
        plt.savefig('results/loss.png', dpi=400)


# 定义训练过程
class Trainer():


    def __init__(self):
        safeCreateDir('results/')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._init_data()
        self._init_model()

    def _init_data(self):
        data = getDataLoader(batch_size=64)
        self.traindl = data.traindl
        self.valdl = data.valdl
        self.testdl = data.testdl
        self.num_words = data.num_words

    def _init_model(self):
        self.textCNN_param = {
            'vocab_size': self.num_words,
            'embed_dim': 64,
            'class_num': 15,
            "kernel_num": 16,
            "kernel_size": [3, 4, 5],
            "dropout": 0.5,
        }
        self.net = textCNN(self.textCNN_param)
        self.opt = Adam(self.net.parameters(), lr=1e-4, weight_decay=5e-4)
        self.cri = nn.CrossEntropyLoss()

    def save_model(self):
        torch.save(self.net.state_dict(), 'saved_dict/cnn.pt')
    def load_model(self):
        self.net.load_state_dict(torch.load('saved_dict/cnn.pt'))


    def train(self,epochs):
        print('init net...')
        self.net.init_weight()
        print(self.net)
        patten = 'Epoch: %d   [===========]  cost: %.2fs;  loss: %.4f;  train acc: %.4f;  val acc:%.4f;'
        train_accs = []
        c_loss = []
        for epoch in range(epochs):
            cur_preds = np.empty(0)
            cur_labels = np.empty(0)
            cur_loss = 0
            start = time.time()
            for batch, (inputs, targets) in enumerate(self.traindl):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                # lengths = lengths.to('cpu')
                self.net.to(self.device)

                pred = self.net(inputs)
                loss = self.cri(pred, targets)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                cur_preds = np.concatenate([cur_preds, pred.cpu().detach().numpy().argmax(axis=1)])

                cur_labels = np.concatenate([cur_labels, targets.cpu().numpy()])
                cur_loss += loss.item()
            # print(cur_preds)
            acc, precision, f1, recall = metrics(cur_preds, cur_labels)
            val_acc, val_precision, val_f1, val_recall = self.val()
            train_accs.append(acc)
            c_loss.append(cur_loss)
            end = time.time()
            print(patten % (epoch+1,end - start,cur_loss, acc,val_acc))

        # self.save_model()
        plot_acc(train_accs)
        plot_loss(c_loss)

    # @torch.no_grad()中的数据不需要计算梯度，也不会进行反向传播
    @torch.no_grad()
    def val(self):
        self.net.eval()
        cur_preds = np.empty(0)
        cur_labels = np.empty(0)
        for batch, (inputs, targets) in enumerate(self.valdl):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            # lengths = lengths.to('cpu')
            self.net.to(self.device)

            pred = self.net(inputs)
            cur_preds = np.concatenate([cur_preds, pred.cpu().detach().numpy().argmax(axis=1)])
            cur_labels = np.concatenate([cur_labels, targets.cpu().numpy()])
        acc, precision, f1, recall = metrics(cur_preds, cur_labels)
        self.net.train()
        return acc, precision, f1, recall
    @torch.no_grad()
    def test(self):
        print("test ...")
        self.load_model()
        patten = 'test acc: %.4f   precision: %.4f   recall: %.4f    f1: %.4f    '
        self.net.eval()
        cur_preds = np.empty(0)
        cur_labels = np.empty(0)
        for batch, (inputs, targets) in enumerate(self.testdl):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            # lengths = lengths.to('cpu')
            self.net.to(self.device)
            # print(self.device)
            pred = self.net(inputs)
            cur_preds = np.concatenate([cur_preds, pred.cpu().detach().numpy().argmax(axis=1)])
            cur_labels = np.concatenate([cur_labels, targets.cpu().numpy()])
        acc, precision, f1, recall = metrics(cur_preds, cur_labels)
        cv_conf = confusion_matrix(cur_preds, cur_labels)
        # print(cv_conf)
        labels11 = ['story','culture','entertainment','sports','finance',
                    'house','car','edu','tech','military',
                    'travel','world','stock','agriculture','game']
        # display_cm(cv_conf,labels11)
        fig, ax = plt.subplots(figsize=(15,15))
        disp = ConfusionMatrixDisplay(confusion_matrix=cv_conf, display_labels=labels11)
        disp.plot(cmap="Blues", values_format='',ax=ax)
        plt.savefig("results/ConfusionMatrix.png", dpi=400)
        self.net.train()
        print(patten % (acc,precision,recall,f1))



if __name__ == "__main__":
    getFormatData() # 数据预处理：数据清洗和词向量
    trainer=Trainer()
    trainer.train(epochs=30) #数据训练
    trainer.test()  # 测试
