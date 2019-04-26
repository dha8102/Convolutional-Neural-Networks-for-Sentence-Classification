### CR input file

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
    precnt = 0
    for i in range(vocalen):
        embeddings.append([0] * 300)

    for i in reverse_voca.keys():
        if reverse_voca[i] in word2vec.keys():
            embeddings[i] = word2vec[reverse_voca[i]]
            precnt += 1
        else:
            for j in range(300):
                # embeddings[i][j] = np.random.randn() / np.sqrt(50/2)
                embeddings[i][j] = np.random.uniform(-0.025, 0.025)
    print("pretrained voca size :", precnt)
    return embeddings


def inputdata():
    data_file = ["Apex AD2600 Progressive-scan DVD player.txt",
                 "Canon G3.txt",
                 "Creative Labs Nomad Jukebox Zen Xtra 40GB.txt",
                 "Nikon coolpix 4300.txt",
                 "Nokia 6610.txt",
                 "Canon PowerShot SD500.txt",
                 "Canon S100.txt",
                 "Diaper Champ.txt",
                 "Hitachi router.txt",
                 "ipod.txt",
                 "Linksys Router.txt",
                 "MicroMP3.txt",
                 "Nokia 6600.txt",
                 "norton.txt"]
    voca = {}
    x = []
    y = []
    vocacnt = 0
    max_sentence_len = 0
    scorelist = ["[-3]", "[-2]", "[-1]", "[+1]", "[+2]", "[+3]"]
    datasize = 0
    # voca 생성
    for thisdata in data_file:
        with open("./dataset/" + thisdata, encoding='latin-1') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line[:-1]

                ifscore = False
                for i, score in enumerate(scorelist):
                    if score in line:
                        ifscore = True
                if ifscore == False: continue
                datasize += 1
                line = line[line.find("##") + 2:]
                cleanline = clean_str(line)
                cleanline = cleanline.split()
                if len(cleanline) > max_sentence_len:
                    max_sentence_len = len(cleanline)
                for i in cleanline:
                    if i not in voca:
                        voca[i] = vocacnt
                        vocacnt += 1
    print("dataset size :", datasize)
    print("voca len :", len(voca))
    print("max sentence length : ", max_sentence_len)
    x_cnt = 0

    for thisdata in data_file:
        with open("./dataset/" + thisdata, encoding='latin-1') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line[:-1]
                ifscore = False
                for i, score in enumerate(scorelist):
                    if score in line:
                        ifscore = True
                        if i <= 2:
                            thisy = [1, 0]
                        else:
                            thisy = [0, 1]
                        break
                if ifscore == False: continue

                x.append([0] * max_sentence_len)
                y.append(thisy)
                line = line[line.find("##") + 2:]
                cleanline = clean_str(line)
                cleanline = cleanline.split()
                wordcnt = 0
                for i in cleanline:
                    x[x_cnt][wordcnt] = voca[i]
                    wordcnt += 1
                x_cnt += 1

    return x, y, voca


if __name__ == "__main__":
    print("CR Data loading...")
    x, y, voca = inputdata()


    # pre-trained vector load
    print("loading pre-trained vector...")
    word2vec = load_pre_train_vector()

    # make pre-trained embeddings
    pre_embeddings = make_embeddings()

    total_data = [x, y, voca, pre_embeddings]
    f = open("./CR_original_data.bin", "wb+")
    pickle.dump(total_data, f)
    f.close()
    for i in range(100, 110):
        print(x[i], y[i])
    print("CR_original_data produced.")
