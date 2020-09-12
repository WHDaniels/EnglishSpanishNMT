import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from tensorflow.python.keras.models import Sequential


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


def preprocess():
    # get lists of both english and spanish phrases
    enPhrases, esPhrases = loadData()

    # tokenizer setup
    tkEn = Tokenizer(char_level=False)
    tkEs = Tokenizer(char_level=False)

    # create index based on word frequency
    tkEn.fit_on_texts(enPhrases)
    tkEs.fit_on_texts(esPhrases)

    # transform index contents to integers
    seqEn = tkEn.texts_to_sequences(enPhrases)
    seqEs = tkEs.texts_to_sequences(esPhrases)

    # post-pad each index to the length of the largest phrase
    padEn = padding(seqEn)
    padEs = padding(seqEs)

    # reshape the second index to be 3-dimensional (for sparse_categorical_crossentropy)
    padEs = padEs.reshape(*padEs.shape, 1)

    # amount of unique words in both languages
    print(str(len(tkEn.word_index)))
    print(str(len(tkEs.word_index)) + "\n\n")

    newEn = padding(seqEn, 49)
    newEn = newEn.reshape((-1, padEs.shape[-2], 1))

    return padEs, tkEs, newEn


def padding(sequence, maxLength=None):
    if maxLength is None:
        # gets the length of the largest phrase through a list comprehension
        maxLength = max([len(phrase) for phrase in sequence])

    # uses kera's padding function to pad zeros to the end of each index
    # up to the length of the largest phrase in that sequence
    paddedSeq = pad_sequences(sequence, maxlen=maxLength, padding="post")

    return paddedSeq


def logitsToText(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


if __name__ == '__main__':
    preEs, tkEs, enInput = preprocess()

    learningRate = 1e-3
    inputSeq = Input(enInput.shape[1:])
    rnn = GRU(64, return_sequences=True)(inputSeq)
    logits = TimeDistributed(Dense(len(tkEs.word_index)))(rnn)
    model = Model(inputSeq, Activation('softmax')(logits))
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learningRate),
                  metrics=['accuracy'])
    model.fit(enInput, preEs, batch_size=1024, epochs=10, validation_split=0.2)

    print(logitsToText(model.predict(enInput[:1][0]), tkEs))
