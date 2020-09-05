import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def loadData():
    enPhrases = []
    esPhrases = []

    # read phrases.txt and add each phrase to its respective list
    with open("phrases.txt", "r", encoding='utf-8') as file:
        content = file.readlines()
        for i in range(0, len(content)):
            line = content[i].split("\t")
            enPhrases.append(line[0])
            esPhrases.append(line[1])

    return enPhrases, esPhrases


def showPhraseStats(enPhrases, esPhrases):
    # FIX THIS LATER IF KEEPING IT
    """
    esWords = 0
    enWords = 0

    # gets the total number of english and spanish words
    for i in range(0, len(enPhrases)):
        enWords += len(enPhrases[i].split())
        esWords += len(esPhrases[i].split())

    nextEn = []
    nextEs = []

    # gets the entire list of unique words (not accurate)
    for j in range(len(enPhrases)):
        for word in enPhrases[j].split():
            if word not in nextEn:
                nextEn.append(word)

        for word in esPhrases[j].split():
            if word not in nextEs:
                nextEs.append(word)

    print(str(enWords) + " English words.\n" + str(len(nextEn)) + " unique English words.\n")
    print(str(esWords) + " Spanish words.\n" + str(len(nextEs)) + " unique Spanish words.\n")
    """


def preprocess():
    # get lists of both english and spanish phrases
    enPhrases, esPhrases = loadData()

    # tokenizer setup
    preEn = Tokenizer(char_level=False)
    preEs = Tokenizer(char_level=False)

    # create index based on word frequency
    preEn.fit_on_texts(enPhrases)
    preEs.fit_on_texts(esPhrases)

    # transform index contents to integers
    seqEn = preEn.texts_to_sequences(enPhrases)
    seqEs = preEs.texts_to_sequences(esPhrases)

    # post-pad each index to the length of the largest phrase
    padEn = padding(seqEn)
    padEs = padding(seqEs)

    # dimensions of the padded arrays
    print(padEn.shape)
    print(padEs.shape)

    # reshape the second index to be 3-dimensional (sparse_categorical_crossentropy requires this)
    padEs = padEs.reshape(*padEs.shape, 1)
    print(padEs.shape)

    # amount of unique words in both languages
    print(len(preEn.word_index))
    print(len(preEs.word_index))


def padding(sequence):
    # gets the length of the largest phrase through a list comprehension
    maxLength = max([len(phrase) for phrase in sequence])

    # uses kera's padding function to pad zeros to the end of each index
    # up to the length of the largest phrase in that sequence
    paddedSeq = pad_sequences(sequence, maxlen=maxLength, padding="post")

    return paddedSeq


if __name__ == '__main__':
    en, es = loadData()
    # showPhraseStats(en, es)
    preprocess()
