# -*- coding: utf-8 -*-
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.models import Model
from keras.layers import  Input, Dense,Embedding,Conv1D,MaxPooling1D,LSTM
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from jieba import lcut
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
from keras.callbacks import TensorBoard,EarlyStopping
import pandas as pd
import PySimpleGUI as sg
import matplotlib

# 数据处理
# """判断一个unicode是否是汉字"""
def is_chinese(uchar):
    if (uchar >= '\u4e00' and uchar <= '\u9fa5') :
        return True
    else:
        return False
def reserve_chinese(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str += i
    return content_str
def getStopWords():
    file = open('./data/stopwords.txt', 'r',encoding='utf8')
    words = [i.strip() for i in file.readlines()]
    file.close()
    return words
def dataParse(text, stop_words):
    label_map = {'news_story': 0, 'news_culture': 1, 'news_entertainment': 2,
               'news_sports': 3, 'news_finance': 4, 'news_house': 5, 'news_car': 6,
               'news_edu': 7, 'news_tech': 8, 'news_military': 9, 'news_travel': 10,
               'news_world': 11, 'stock': 12, 'news_agriculture': 13, 'news_game': 14}
    _, _, label, content, _ = text.split('_!_')
    label = label_map[label]
    content = reserve_chinese(content)
    words = lcut(content)
    words = [i for i in words if not i in stop_words]
    return words, int(label)

def getData(file='./data/toutiao_cat_data.txt',):
    file = open(file, 'r',encoding='utf8')
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
    return all_words,all_labels

## 读取测数据集
data,label = getData()

X_train, X_t, train_y, v_y = train_test_split(data,label,test_size=0.3, random_state=42)
X_val, X_test, val_y, test_y = train_test_split(X_t,v_y,test_size=0.5, random_state=42)
# print(X_train)

## 对数据集的标签数据进行one-hot编码
ohe = OneHotEncoder()
train_y = ohe.fit_transform(np.array(train_y).reshape(-1,1)).toarray()
val_y = ohe.transform(np.array(val_y).reshape(-1,1)).toarray()
test_y = ohe.transform(np.array(test_y).reshape(-1,1)).toarray()
print('训练集',train_y.shape)
print('验证集',val_y.shape)
print('测试集',test_y.shape)

## 使用Tokenizer对词组进行编码
## 当我们创建了一个Tokenizer对象后，使用该对象的fit_on_texts()函数，以空格去识别每个词,
## 可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小。
max_words = 5000
max_len = 100
tok = Tokenizer(num_words=max_words)  ## 使用的最大词语数为5000
tok.fit_on_texts(data)

# texts_to_sequences 输出的是根据对应关系输出的向量序列，是不定长的，跟句子的长度有关系
print(X_train[0])
train_seq = tok.texts_to_sequences(X_train)
val_seq = tok.texts_to_sequences(X_val)
test_seq = tok.texts_to_sequences(X_test)
print(train_seq[0])
## 将每个序列调整为相同的长度.长度为100
train_seq_mat = pad_sequences(train_seq,maxlen=max_len)
val_seq_mat = pad_sequences(val_seq,maxlen=max_len)
test_seq_mat = pad_sequences(test_seq,maxlen=max_len)

num_classes = 15
## 定义CNN-LSTM模型
inputs = Input(name='inputs',shape=[max_len])
## Embedding(词汇表大小,batch大小,每个新闻的词长)
layer = Embedding(max_words+1,128,input_length=max_len)(inputs)
layer = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(layer)
layer = MaxPooling1D(pool_size=2)(layer)
layer = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(layer)
layer = MaxPooling1D(pool_size=2)(layer)
layer = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(layer)
layer = Dense(num_classes, activation='softmax')(layer)
model = Model(inputs=inputs,outputs=layer)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# #模型训练

model.fit(train_seq_mat,train_y,batch_size=128,epochs=10,
                      validation_data=(val_seq_mat,val_y),
                      callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001),TensorBoard(log_dir='./log')]
                        ## 当val-loss不再降低时停止训练
                     )
# # 保存模型
model.save('model/CNN-LSTM.h5')
del model

## 对验证集进行预测
# 导入已经训练好的模型
model = load_model('model/CNN-LSTM.h5')
test_pre = model.predict(test_seq_mat)
pred = np.argmax(test_pre,axis=1)
real = np.argmax(test_y,axis=1)
cv_conf = confusion_matrix(real, pred)
acc = accuracy_score(real, pred)
precision = precision_score(real, pred, average='micro')
recall = recall_score(real, pred, average='micro')
f1 = f1_score(real, pred, average='micro')
patten = 'test:  acc: %.4f   precision: %.4f   recall: %.4f   f1: %.4f'
print(patten % (acc,precision,recall,f1,))
labels11 = ['story','culture','entertainment','sports','finance',
                    'house','car','edu','tech','military',
                    'travel','world','stock','agriculture','game']
fig, ax = plt.subplots(figsize=(15,15))
disp = ConfusionMatrixDisplay(confusion_matrix=cv_conf, display_labels=labels11)
disp.plot(cmap="Blues", values_format='',ax=ax)
plt.savefig("ConfusionMatrix.tif", dpi=400)

def dataParse_(content, stop_words):
    content = reserve_chinese(content)
    words = lcut(content)
    words = [i for i in words if not i in stop_words]
    return words
def getData_one(file):
    file = open(file, 'r',encoding='utf8')
    texts = file.readlines()
    file.close()
    stop_words = getStopWords()
    all_words = []
    word = []
    for text in texts:
        content = dataParse_(text, stop_words)
        if len(content) <= 0:
            continue
        all_words.append(content)
        word.append(text)
    return all_words,word

def predict_(file):
    data_cut ,data= getData_one(file)
    t_seq = tok.texts_to_sequences(data_cut)
    t_seq_mat = pad_sequences(t_seq, maxlen=max_len)
    model = load_model('model/CNN-LSTM.h5')
    t_pre = model.predict(t_seq_mat)
    pred = np.argmax(t_pre, axis=1)
    labels11 = ['story', 'culture', 'entertainment', 'sports', 'finance',
                'house', 'car', 'edu', 'tech', 'military',
                'travel', 'world', 'stock', 'agriculture', 'game']
    pred_lable = []
    for i in pred:
        pred_lable.append(labels11[i])
    df_x = pd.DataFrame(data)
    df_y = pd.DataFrame(pred_lable)
    headerList = ['label', 'text']
    data = pd.concat([df_y, df_x], axis=1)
    data.to_csv('data.csv',header=headerList,)
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    data = pd.read_csv('data.csv')
    result = data['label'].value_counts()
    result.plot(kind='bar')
    plt.show()
    # return pred_lable

def main_windows():
    # 菜单栏
    menu_def = [['Help', 'About...'], ]
    # 主窗口
    layout = [[sg.Menu(menu_def, tearoff=True)],
              [sg.Text('')],
              [sg.Text('请选择要处理的文本',font=("Helvetica", 16)),],
              [sg.Text('导入文本', size=(8, 1),font=("Helvetica", 16)), sg.Input(), sg.FileBrowse()],
              [sg.Text('')],
              [sg.Text('', size=(20, 1)), sg.Button('启动数据处理',font=("Helvetica", 16))],
              [sg.Text('')],
              [sg.Text('', size=(20, 1)), sg.Text(key="output", justification='center',font=("Helvetica", 16))],
              [sg.Text('')],
              [sg.Text('')], ]
    win1 = sg.Window('中文文本分类系统', layout)
    while True:
        ev1, vals1 = win1.Read()
        if ev1 is None:
            break
        if ev1 == '启动数据处理':
            predict_(vals1[1])
            win1['output'].update('处理完毕')
        else:
            pass

if __name__ == "__main__":
    main_windows()
