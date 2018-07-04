# ref : https://blog.codingecho.com/2018/03/25/lstm%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%A6%E3%83%86%E3%82%AD%E3%82%B9%E3%83%88%E3%81%AE%E5%A4%9A%E3%82%AF%E3%83%A9%E3%82%B9%E5%88%86%E9%A1%9E%E3%82%92%E3%81%99%E3%82%8B/
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding, Input, concatenate
from keras.utils import to_categorical
from keras.layers import LSTM
from keras.models import Model
from keras import backend as K
import MeCab

batch_size=1

v=np.array(
[[1,0,0,0,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0,0,0],
[0,0,1,0,0,0,0,0,0,0],
[0,0,0,1,0,0,0,0,0,0],
[0,0,0,0,1,0,0,0,0,0],
[0,0,0,0,0,1,0,0,0,0],
[0,0,0,0,0,0,1,0,0,0],
[0,0,0,0,0,0,0,1,0,0],
[1,0,0,0,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0,0,0],
[0,0,1,0,0,0,0,0,0,0],
[0,0,0,1,0,0,0,0,0,0],
[0,0,0,0,0,0,0,1,0,0],
[0,0,0,1,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,1,0],
[0,0,0,0,0,0,0,0,0,1],
[1,0,0,0,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0,0,0],
[0,0,1,0,0,0,0,0,0,0],
[0,0,0,1,0,0,0,0,0,0],
[0,0,0,0,0,0,0,1,0,0],
[0,0,0,1,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,1,0],
[0,0,0,0,0,0,0,0,0,1]]
)
print(v.shape)

#l=np.array([[2,3,5]])
l=np.array([[2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,5,5,5,5,5,5,5,5]])
l=to_categorical(l, num_classes=6)
l

"""
l1=np.array([[2,2,2,2,2,2,2,2]])
l2=np.array([[3,3,3,3,3,3,3,3]])
l3=np.array([[5,5,5,5,5,5,5,5]])

l1=to_categorical(l1, num_classes=6)
l2=to_categorical(l2, num_classes=6)
l3=to_categorical(l3, num_classes=6)

l=np.append(l,[l1])
l=np.append(l,[l2])
l=np.append(l,[l3])
"""
print(l.shape)
nb_label=l.shape[1]

def loaddata(path=None):
    with open(path, encoding="utf-8") as f:
        #l_strip = [s.strip().split("。") for s in f.readlines() if len(s)>1]
        l_strip = [s.split("。") for s in f.readlines() if len(s)>1]
        l_strip = [s for snt in l_strip for s in snt if s!="\n"]
        #print(l_strip)
    return l_strip[0:3]

def morpheme(textlist):
    mecab = MeCab.Tagger ('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    morph_textlist = []
    for sentence in textlist:
        sentence = mecab.parse(sentence)
        sentence = sentence.split('\n')[:-2]
        words = []
        for sentence in sentence:
            words.append(sentence.split('\t')[0]) # 形態素解析したあとの1文内の単語の集合
        morph_textlist.append(words)
    # ストップワードの除去
    # https://qiita.com/Hironsan/items/2466fe0f344115aff177#%E3%82%B9%E3%83%88%E3%83%83%E3%83%97%E3%83%AF%E3%83%BC%E3%83%89%E3%81%AE%E9%99%A4%E5%8E%BB
    return morph_textlist

def recall_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f_score(y_true, y_pred):
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    return 2 * pre * rec / (pre + rec)

def LSTM_classify(batch_size, maxlen, vocab):

    n_dim=2

    inputs=Input(shape=(vocab,), dtype='int32')
    #inputs=Input(shape=(maxlen,), dtype='int32')
    print("input shape : ", inputs.shape)

    #x=Embedding(output_dim=n_dim, input_dim=vocab, input_length=maxlen)(inputs)
    x=Embedding(output_dim=n_dim, input_dim=vocab, input_length=vocab)(inputs)
    print("x.shape:",x.shape)
    x=LSTM(32, input_shape=(maxlen, n_dim), return_sequences=False)(x)
    print("x.shape:",x.shape)
    #x=Flatten()(x)
    x=Dense(nb_label, activation='softmax')(x)
    print("x.shape:",x.shape)

    model = Model(inputs=[inputs], outputs=[x])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', recall_score, precision_score, f1_score])
    model.summary()

    return model

def LSTM_classify1(batch_size, maxlen, vocab):

    n_dim=2

    model=Sequential()
    model.add(Embedding(output_dim=n_dim, input_dim=vocab, input_length=vocab))
    model.add(LSTM(32, input_shape=(maxlen, n_dim), return_sequences=False))
    #x=Flatten()(x)
    model.add(Dense(nb_label, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', recall_score, precision_score, f1_score])
    print(model.summary())

    return model

def main():

    train_data, train_label, test_data, test_label = v, l, v, l
    print("train_data", train_data.shape)
    print("train_label", train_label.shape)
    #maxlen, vocab = train_data.shape[1], train_data.shape[2]
    maxlen, vocab=8, train_data.shape[1]

    print("batch_size", batch_size, "maxlen", maxlen, "vocab", vocab)

    model = LSTM_classify(batch_size, maxlen, vocab)

    history = model.fit(train_data, train_label, verbose=1, epochs=32, batch_size=batch_size, validation_data=(test_data, test_label), shuffle=False)
    """
    epochs=32
    for _ in range(epochs):
        for i in range(train_data.shape[0]):
            history = model.fit(train_data[i], train_label[i], verbose=1, epochs=32, batch_size=batch_size, validation_data=(test_data[i], test_label[i]), shuffle=False)
    """
#main()
#for i in range(1,10):
#    loaddata(path="./data/000"+str(i)+".txt")
text=loaddata(path="./data/0002.txt")
print(text)
print(morpheme(text))

"""
    sentence=[]

    x1=Embedding(output_dim=n_dim, input_dim=vocab, input_length=maxlen)(inputs[0])
    x2=Embedding(output_dim=n_dim, input_dim=vocab, input_length=maxlen)(inputs[1])
    x3=Embedding(output_dim=n_dim, input_dim=vocab, input_length=maxlen)(inputs[2])
    x4=Embedding(output_dim=n_dim, input_dim=vocab, input_length=maxlen)(inputs[3])
    x5=Embedding(output_dim=n_dim, input_dim=vocab, input_length=maxlen)(inputs[4])
    x6=Embedding(output_dim=n_dim, input_dim=vocab, input_length=maxlen)(inputs[5])
    x7=Embedding(output_dim=n_dim, input_dim=vocab, input_length=maxlen)(inputs[6])
    x8=Embedding(output_dim=n_dim, input_dim=vocab, input_length=maxlen)(inputs[7])
    print(type(x1))

    x=concatenate([x1,x2,x3,x4,x5,x6,x7,x8], axis=2)
"""
