from random import shuffle

import numpy as np
from keras import callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.saving.save import save_model


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
    # shuffle the data
    shuffle(enPhrases), shuffle(esPhrases)

    return enPhrases, esPhrases


def preprocess():
    # get lists of both english and spanish phrases
    enPhrases, esPhrases = loadData()

    # tokenizer setup
    tkEn = Tokenizer(char_level=False, filters='¡¿!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tkEs = Tokenizer(char_level=False, filters='¡¿!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')

    # create index based on word frequency
    tkEn.fit_on_texts(enPhrases)
    tkEs.fit_on_texts(esPhrases)

    # transform index contents to integers
    seqEn = tkEn.texts_to_sequences(enPhrases)
    seqEs = tkEs.texts_to_sequences(esPhrases)

    # post-pad spanish sequence to the length of the largest phrase in that list
    padEs = padding(seqEs)

    # reshape the second index to be 3-dimensional (for sparse_categorical_crossentropy)
    padEs = padEs.reshape(*padEs.shape, 1)

    # amount of unique words in both languages
    # print(str(len(tkEn.word_index)))
    # print(str(len(tkEs.word_index)) + "\n\n")

    # pad and reshape the input so it can be used in the fit function
    # newEn = padding(seqEn, padEs.shape[1])
    newEn = padding(seqEn)
    # newEn = newEn.reshape((-1, padEs.shape[-2]))

    return padEs, tkEn, tkEs, newEn


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
    preEs, tkEn, tkEs, enInput = preprocess()

    learningRate = 1e-5
    model = Sequential()

    # model.add(Embedding(len(tkEs.word_index), 64, input_length=enInput.shape[1]))
    # model.add(GRU(64, return_sequences=True, activation="tanh"))
    # model.add(TimeDistributed(Dense(len(tkEs.word_index), activation="softmax")))

    model.add(Embedding(input_dim=len(tkEn.word_index)+1, output_dim=128, input_length=enInput.shape[1]))
    model.add(Bidirectional(GRU(256, return_sequences=False)))
    model.add(RepeatVector(preEs.shape[1]))
    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add(TimeDistributed(Dense(len(tkEs.word_index)+1, activation="softmax")))

    # compiles model
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learningRate),
                  metrics=['accuracy'])

    # callback to save best model
    checkpointPath = 'C:\\Users\\mercm\\OneDrive\\Documents\\GitHub\\EnglishSpanishNMT\\model'
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpointPath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # trains model
    model.fit(enInput, preEs, batch_size=32, epochs=40, validation_split=0.2, shuffle=True,
              callbacks=[model_checkpoint_callback])

    # saves model after training
    # savePath = 'C:\\Users\\mercm\\OneDrive\\Documents\\GitHub\\EnglishSpanishNMT\\model'
    # save_model(model, savePath)

    # print(sequenceToText(model.predict(enInput[:1])[0], tkEs))
