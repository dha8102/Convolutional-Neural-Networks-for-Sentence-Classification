

### SST2 input file

import pickle
import re
import numpy as np
from gensim.models import KeyedVectors


def clean_str(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


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

    # pre-trained vector 분산의 평균 구하기
    embed_var = 0
    for i in word2vec.keys():
        embed_var += np.var(word2vec[i])
    embed_var /= len(word2vec.keys())

    for i in range(vocalen):
        embeddings.append([0] * 300)
    precnt = 0
    for i in reverse_voca.keys():
        if reverse_voca[i] in word2vec.keys():
            embeddings[i] = word2vec[reverse_voca[i]]
            precnt += 1
        else:
            for j in range(300):
                embeddings[i][j] = np.random.uniform(-0.01, 0.01)
    print("embeddings_len:", len(embeddings))
    print("precnt:", precnt)
    return embeddings


def y_append(this_sent):
    if this_sent == 0:
        return [1, 0]
    elif this_sent == 1:
        return [0, 1]



def inputdata():
    file = ["./dataset/stsa.binary.train",
                "./dataset/stsa.binary.test",
                "./dataset/stsa.binary.dev"]
    # voca 생성
    max_sentence_len = 0
    voca = {}
    vocacnt = 0
    sentencecnt = 0
    sentence = []
    cnt = 0
    for filename in file:
        with open(filename, encoding='latin-1') as f:
            while True:
                line = f.readline()
                if not line: break
                sentencecnt += 1
                cleanline = line[2:-1]
                sentence.append(cleanline)
                cleanline = clean_str(cleanline)
                cleanline = cleanline.split()
                if len(cleanline) > max_sentence_len:
                    max_sentence_len = len(cleanline)
                for i in cleanline:
                    if i not in voca:
                        voca[i] = vocacnt
                        vocacnt += 1

    print("voca len :", len(voca))
    print("max sentence len : ", max_sentence_len)
    print("Total number of whole sentences : ", sentencecnt)

    x_train = []
    y_train = []
    x_cnt = 0
    filename = "./dataset/stsa.binary.train"
    with open(filename, encoding='latin-1') as f:
        while True:
            line = f.readline()
            if not line:
                break
            x_train.append([0] * max_sentence_len)
            thisy = int(line.split(" ")[0])
            y_train.append(y_append(thisy))
            cleanline = line[2:-1]
            cleanline = clean_str(cleanline)
            cleanline = cleanline.split()
            wordcnt = 0
            for i in cleanline:
                x_train[x_cnt][wordcnt] = voca[i]
                wordcnt += 1
            x_cnt += 1

    x_test = []
    y_test = []
    x_cnt = 0
    filename = "./dataset/stsa.binary.test"
    with open(filename, encoding='latin-1') as f:
        while True:
            line = f.readline()
            if not line:
                break
            x_test.append([0] * max_sentence_len)
            thisy = int(line.split(" ")[0])
            y_test.append(y_append(thisy))
            cleanline = line[2:-1]
            cleanline = clean_str(cleanline)
            cleanline = cleanline.split()
            wordcnt = 0
            for i in cleanline:
                x_test[x_cnt][wordcnt] = voca[i]
                wordcnt += 1
            x_cnt += 1

    x_dev = []
    y_dev = []
    x_cnt = 0
    filename = "./dataset/stsa.binary.dev"
    with open(filename, encoding='latin-1') as f:
        while True:
            line = f.readline()
            if not line:
                break
            x_dev.append([0] * max_sentence_len)
            thisy = int(line.split(" ")[0])
            y_dev.append(y_append(thisy))
            cleanline = line[2:-1]
            cleanline = clean_str(cleanline)
            cleanline = cleanline.split()
            wordcnt = 0
            for i in cleanline:
                x_dev[x_cnt][wordcnt] = voca[i]
                wordcnt += 1
            x_cnt += 1
    return sentence, x_train, x_test, x_dev, y_train, y_test, y_dev, voca


if __name__ == "__main__":
    print("SST Data loading...")
    # making x
    sentence, x_train, x_test, x_dev, y_train, y_test, y_dev, voca = inputdata()

    print("x len:", len(x_train) + len(x_test) + len(x_dev))
    print("y len:", len(y_train) + len(y_test) + len(y_dev))
    # pre-trained vector load
    print("loading pre-trained vector...")
    word2vec = load_pre_train_vector()

    # make pre-trained embeddings
    pre_embeddings = make_embeddings()

    total_data = [x_train, x_test, x_dev, y_train, y_test, y_dev, voca, pre_embeddings]
    f = open("./SST2_original_data.bin", "wb+")
    pickle.dump(total_data, f)
    f.close()
    print("SST2_original_data produced.")
