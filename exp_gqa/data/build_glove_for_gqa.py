import numpy as np


def initEmbRandom(num, dim):
    # uniform initialization
    wrdEmbScale = 1.
    lowInit = -1.0 * wrdEmbScale
    highInit = 1.0 * wrdEmbScale
    embeddings = np.random.uniform(
        low=lowInit, high=highInit, size=(num, dim))
    return embeddings


wordVectorsFile = './glove/glove.6B.300d.txt'
wordVectors = {}
with open(wordVectorsFile, "r") as inFile:
    for line in inFile:
        line = line.strip().split()
        word = line[0].lower()
        vector = np.array([float(x) for x in line[1:]])
        wordVectors[word] = vector

with open('./vocabulary_gqa.txt') as f:
    vocab_list = [w.strip() for w in f.readlines()]

np.random.seed(3)
embeddings = initEmbRandom(len(vocab_list), 300)

for idx, w in enumerate(vocab_list):
    if w in wordVectors:
        embeddings[idx] = wordVectors[w]
    else:
        print('{} is not in GloVe dict'.format(w))

np.save('./glove_gqa.npy', embeddings)
