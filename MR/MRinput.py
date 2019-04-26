
### MR input file

import pickle
import re
import numpy as np
import random
from gensim.models import KeyedVectors


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def load_pre_train_vector():
    embedding_index = {}
    filepath = "/hdd/data/text/pretrained_vector/GoogleNews-vectors-negative300.bin"
    wv_from_bin = KeyedVectors.load_word2vec_format(filepath, binary=True)
    for word, vector in zip(wv_from_bin.vocab, wv_from_bin.vectors):
        coefs = np.asarray(vector, dtype='float32')
        embedding_index[word] = coefs
    return embedding_index


def make_embeddings():
    reverse_voca = {}
    for i in voca.keys():
        reverse_voca[voca[i]] = i

    vocalen = max(voca.values()) + 1
    embeddings = []

    for i in range(vocalen):
        embeddings.append([0] * 300)

    for i in reverse_voca.keys():
        if reverse_voca[i] in word2vec.keys():
            embeddings[i] = word2vec[reverse_voca[i]]
        else:
            for j in range(300):
                # embeddings[i][j] = np.random.randn() / np.sqrt(50/2)
                embeddings[i][j] = np.random.uniform(-0.025, 0.025)

    return embeddings


def inputdata():
    data_file = ["rt-polarity.pos", "rt-polarity.neg"]
    voca = {}
    x = []
    y = []
    vocacnt = 0
    max_sentence_len = 0
    # voca 생성
    for thisdata in data_file:
        with open(thisdata, encoding='latin-1') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                cleanline = clean_str(line)
                cleanline = cleanline.split()
                if len(cleanline) > max_sentence_len:
                    max_sentence_len = len(cleanline)
                for i in cleanline:
                    if i not in voca:
                        voca[i] = vocacnt
                        vocacnt += 1

    print("voca len :", len(voca))
    print("max sentence len : ", max_sentence_len)
    x_cnt = 0

    with open("rt-polarity.pos", encoding="latin-1") as f:
        while True:
            line = f.readline()
            if not line:
                break
            x.append([0] * max_sentence_len)
            y.append([1, 0])
            cleanline = clean_str(line)
            cleanline = cleanline.split()
            wordcnt = 0
            for i in cleanline:
                x[x_cnt][wordcnt] = voca[i]
                wordcnt += 1
            x_cnt += 1

    with open("rt-polarity.neg", encoding="latin-1") as f:
        while True:
            line = f.readline()
            if not line:
                break
            x.append([0] * max_sentence_len)
            y.append([0, 1])
            cleanline = clean_str(line)
            cleanline = cleanline.split()
            wordcnt = 0
            for i in cleanline:
                x[x_cnt][wordcnt] = voca[i]
                wordcnt += 1
            x_cnt += 1

    return x, y, voca


if __name__ == "__main__":
    print("MR Data loading...")
    x, y, voca = inputdata()
    for i in range(10000, 10010):
        print(x[i], y[i])

    # pre-trained vector load
    print("loading pre-trained vector...")
    word2vec = load_pre_train_vector()

    # make pre-trained embeddings
    pre_embeddings = make_embeddings()

    total_data = [x, y, voca, pre_embeddings]
    f = open("./MR_original_data.bin", "wb+")
    pickle.dump(total_data, f)
    f.close()
    print("MR_original_data produced.")
