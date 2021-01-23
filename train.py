import os
import random
import re
from collections import Counter, OrderedDict

import numpy as np
from keras import callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam, RMSprop
from keras.losses import sparse_categorical_crossentropy
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.saving.save import save_model


def combineData():
    """
    with open('TatoebaEN.txt', 'r', encoding='utf-8') as fileEN, \
            open('TatoebaES.txt', 'r', encoding='utf-8') as fileES, \
            open('Tatoeba.txt', 'w', encoding='utf-8') as file:
        EN, ES = fileEN.readlines(), fileES.readlines()
        for i in range(len(EN)):
            file.write(EN[i].rstrip("\n") + "\t" + ES[i])
    """
    """
    with open("TatoebaEN.txt", 'r', encoding='utf-8') as en, \
            open("TatoebaES.txt", 'r', encoding='utf-8') as es, \
            open('phrases.txt', 'r', encoding='utf-8') as phr, \
            open("combinedSmall.txt", 'w', encoding='utf-8') as target:
        enRead, esRead, phrRead = en.readlines(), es.readlines(), phr.readlines()
    
    
        for i in phrRead:
            target.write(i)

        # first n lines (as opposed to len(enRead) lines)
        n = 50000
        # n = len(enRead)
        for j in range(n):
            newLine = enRead[j].rstrip("\n") + "\t" + esRead[j]
            # if the new line is less than 1024 characters long,
            # contains\starts with a letter, and doesn't contain numbers
            if len(newLine) < 1024 and re.search('[a-zA-Z]', newLine) and re.search('[a-zA-Z]', newLine[0]) and \
                    not re.search('[0-9]', newLine):
                target.write(enRead[j].rstrip("\n") + "\t" + esRead[j])
    """
    with open("small_vocab_en.txt", "r", encoding='utf-8') as en,\
        open("small_vocab_fr.txt", "r", encoding='utf-8') as es,\
        open("new.txt", "w", encoding='utf-8') as new:

        enRead, esRead = en.readlines(), es.readlines()

        for j in range(len(enRead)):
            new.write(enRead[j].rstrip("\n").replace('.', '') + "\t" + esRead[j].replace('.', ''))


def loadData():
    enPhrases = []
    esPhrases = []

    # read phrase file and add each phrase to its respective list
    translationFile = "finalReduced6.txt"
    with open(translationFile, "r", encoding='utf-8') as file:
        content = file.readlines()

        print("Shuffling data")
        # shuffle the data before adding to list
        random.shuffle(content)
        print("Done Shuffling")

        # how many phrase couples to train on
        # phraseCoupleCount = 50000

        for i in range(0, len(content[:])):
            line = content[i].split("\t")
            enPhrases.append(line[0])
            esPhrases.append(line[1])
            # print(line[0], " <---> ", line[1])

    return enPhrases, esPhrases


# returns a cutoff value based off of n
def firstWordOfFreqN(n, phraseList):
    # tk = Tokenizer(char_level=False, filters='¡¿!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tk = Tokenizer()
    tk.fit_on_texts(phraseList)

    newDict = OrderedDict({k: v for k, v in sorted(tk.word_counts.items(), key=lambda item: item[1])})

    print("word_count: {}".format(len(tk.word_counts.items())))

    wordDifference = 0
    for i in reversed(newDict):
        wordDifference += 1
        if newDict[i] <= n:
            return wordDifference


def preprocess():
    # makes a txt file to load data from (only needs to be done once)
    # combineData()

    # get lists of both english and spanish phrases
    enPhrases, esPhrases = loadData()

    """
    print("Test loadData")
    for x in range(len(enPhrases)):
        print(enPhrases[x], " <--> ", esPhrases[x])

    print("num entries: ", len(enPhrases), len(esPhrases))
    """

    # word frequency cutoff
    n = 3

    englishNumWords = firstWordOfFreqN(n, enPhrases)
    spanishNumWords = firstWordOfFreqN(n, esPhrases)

    print("Actual english vocab: " + str(englishNumWords))
    print("Actual spanish vocab: " + str(spanishNumWords))

    # tokenizer setup
    """
    tkEn = Tokenizer(num_words=englishNumWords + 2, oov_token="<UNK>", char_level=False,
                     filters='¡¿!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tkEs = Tokenizer(num_words=spanishNumWords + 2, oov_token="<UNK>", char_level=False,
                     filters='¡¿!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    """
    tkEn = Tokenizer()
    tkEs = Tokenizer()

    # create index based on word frequency
    tkEn.fit_on_texts(enPhrases)
    tkEs.fit_on_texts(esPhrases)

    print("word_count_en: {}".format(len(tkEn.word_counts.items())))
    print("word_count_es: {}".format(len(tkEs.word_counts.items())))

    print("tkEn.word_index: {}".format(len(tkEn.word_index)))
    print("tkEs.word_index: {}".format(len(tkEs.word_index)))

    # transform index contents to integers
    seqEn = tkEn.texts_to_sequences(enPhrases)
    seqEs = tkEs.texts_to_sequences(esPhrases)

    # print(seqEn)
    # print(seqEs)

    # post-pad english/spanish sequence to the length of the largest phrase in that list
    padEn = padding(seqEn)
    padEs = padding(seqEs)

    # print("padEn.shape: {}".format(padEn.shape))
    # print("padEs.shape: {}".format(padEs.shape))

    # reshape the second index to be 3-dimensional (for sparse_categorical_crossentropy)
    padEs = padEs.reshape(*padEs.shape, 1)

    # amount of unique words in both languages
    # print(str(len(tkEn.word_index)))
    # print(str(len(tkEs.word_index)) + "\n\n")

    # pad and reshape the input so it can be used in the fit function
    # newEn = padding(seqEn, padEs.shape[1])
    newEn = padding(seqEn)
    # newEn = newEn.reshape((-1, padEs.shape[-2]))

    print("Preprocessing finished\n")
    return padEn, padEs, tkEn, tkEs, newEn


def padding(sequence, maxLength=None):
    if maxLength is None:
        # if the maxLength (word length of largest phrase) is not specified
        # gets the length of the largest phrase through a list comprehension
        maxLength = max([len(phrase) for phrase in sequence])

    # uses kera's padding function to pad zeros to the end of each index
    # up to the length of the largest phrase in that sequence
    paddedSeq = pad_sequences(sequence, maxlen=maxLength, padding="post")

    return paddedSeq


def sequenceToText(sequence, tk):
    sequenceToWords = {wordID: word for word, wordID in tk.word_index.items()}
    sequenceToWords[0] = '<PAD>'

    return ' '.join([sequenceToWords[prediction] for prediction in np.argmax(sequence, 1)])


if __name__ == '__main__':
    # combineData()
    # combined = open('combined.txt', 'r', encoding='utf-8')

    preEn, padEs, tkEn, tkEs, enInput = preprocess()

    print("English Vocab: {}".format(len(tkEn.word_index)))
    print("\nSpanish Vocab: {}".format(len(tkEs.word_index)))

    # learningRate = 0.003
    learningRate = 0.005
    model = Sequential()

    # add layers
    model.add(Embedding(input_dim=len(tkEn.word_index) + 1, output_dim=128, input_length=enInput.shape[1]))
    model.add(Bidirectional(GRU(256, return_sequences=False)))
    model.add(RepeatVector(padEs.shape[1]))
    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add(TimeDistributed(Dense(len(tkEs.word_index) + 1, activation="softmax")))

    # compiles model
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learningRate),
                  metrics=['accuracy'])

    # callback to save best model
    checkpointPath = (os.getcwd() + '\\finalReducedModel2')
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpointPath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # trains model
    model.fit(enInput, padEs, batch_size=64, epochs=50, validation_split=0.2, callbacks=[model_checkpoint_callback])

    # saves model after training
    # savePath = 'C:\\Users\\mercm\\OneDrive\\Documents\\GitHub\\EnglishSpanishNMT\\model'
    # save_model(model, savePath)

    # print(sequenceToText(model.predict(enInput[:1])[0], tkEs))
