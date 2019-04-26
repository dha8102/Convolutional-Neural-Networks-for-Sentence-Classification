### TREC input file

import pickle
import re
import numpy as np
import random
from gensim.models import KeyedVectors


def clean_str(string, TREC=True):
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
    # return string.strip().lower()

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
                embeddings[i][j] = np.random.uniform(-0.25, 0.25)

    return embeddings


def y_one_hot(this_sent):
    if this_sent == "DESC":
        return [1, 0, 0, 0, 0, 0]
    elif this_sent == "ENTY":
        return [0, 1, 0, 0, 0, 0]
    elif this_sent == "ABBR":
        return [0, 0, 1, 0, 0, 0]
    elif this_sent == "HUM":
        return [0, 0, 0, 1, 0, 0]
    elif this_sent == "LOC":
        return [0, 0, 0, 0, 1, 0]
    else:
        return [0, 0, 0, 0, 0, 1]


def inputdata():
    data_file = ["./dataset/traindata.txt", "./dataset/testdata.txt"]
    voca = {}
    x_train = []
    y_train = []
    vocacnt = 0
    max_sentence_len = 0
    # voca 생성
    for thisdata in data_file:
        with open(thisdata, encoding='UTF-8') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line[:-1]
                cleanline = clean_str(line)
                cleanline = cleanline.split()
                cleanline = cleanline[2:]
                if len(cleanline) > max_sentence_len:
                    max_sentence_len = len(cleanline)
                for i in cleanline:
                    if i not in voca:
                        voca[i] = vocacnt
                        vocacnt += 1

    print("voca 길이 :", len(voca))
    print("max 문장 길이 : ", max_sentence_len)
    x_cnt = 0

    with open("./dataset/traindata.txt", encoding="UTF-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line[:-1]
            x_train.append([0] * max_sentence_len)

            cleanline = clean_str(line)
            cleanline = cleanline.split()


            y_train.append(y_one_hot(cleanline[0]))

            cleanline = cleanline[2:]
            wordcnt = 0
            for i in cleanline:
                x_train[x_cnt][wordcnt] = voca[i]
                wordcnt += 1
            x_cnt += 1

    x_test = []
    y_test = []
    x_cnt = 0
    with open("./dataset/testdata.txt", encoding="UTF-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line[:-1]
            x_test.append([0] * max_sentence_len)

            cleanline = clean_str(line)
            cleanline = cleanline.split()
            y_test.append(y_one_hot(cleanline[0]))

            cleanline = cleanline[2:]
            wordcnt = 0
            for i in cleanline:
                x_test[x_cnt][wordcnt] = voca[i]
                wordcnt += 1
            x_cnt += 1

    return x_train, x_test, y_train, y_test, voca


if __name__ == "__main__":
    print("TREC Data loading...")
    x_train, x_test, y_train, y_test, voca = inputdata()
    for i in range(10):
        print(x_train[i], y_train[i])


    # pre-trained vector load
    print("loading pre-trained vector...")
    word2vec = load_pre_train_vector()

    # make pre-trained embeddings
    pre_embeddings = make_embeddings()

    total_data = [x_train, x_test, y_train, y_test, voca, pre_embeddings]
    f = open("./TREC_original_data.bin", "wb+")
    pickle.dump(total_data, f)
    f.close()
    print("TREC_original_data produced.")
