import pickle
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from tensorflow import keras
import os

model = keras.models.load_model(os.getcwd() + '\\FINAL1')
padEn, padEs = pickle.load(open('data//padEn.p', 'rb')), pickle.load(open('data//padEs.p', 'rb'))
tkEn, tkEs = pickle.load(open('data//tkEn.p', 'rb')), pickle.load(open('data//tkEs.p', 'rb'))


def separatePhrases(testFile):
    enPhrases, esPhrases = list(), list()

    with open(testFile, "r", encoding="utf-8") as test:
        testRead = test.readlines()

        for n, pairing in enumerate(testRead):
            separate = pairing.split("\t")
            enPhrases.append(separate[0])
            esPhrases.append(separate[1])

    return enPhrases, esPhrases


def testAccuracy(enPhrases, esPhrases):
    translatedList = []

    for n, phrase in enumerate(enPhrases):
        # mirrors the key to value relationship in tkEn.word_index
        y_id_to_word = {value: key for key, value in tkEs.word_index.items()}
        y_id_to_word[0] = '<PAD>'

        original = enPhrases[n]
        sentence = ''.join(c for c in enPhrases[n] if c not in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n').lower()
        # print(sentence)

        # sentence equals the list of ids that correspond to each word in sentence
        sentence = [tkEn.word_index[word] for word in sentence.split()]

        # post pad the sentence to the length of the n dimension
        sentence = pad_sequences([sentence], maxlen=padEn.shape[-1], padding='post')
        sentences = np.array([sentence[0], padEn[0]])

        predictions = model.predict(sentences, len(sentences))

        print(original)
        translated = ' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]
                               if y_id_to_word[np.argmax(x)] != "<PAD>"])
        print(translated)
        translatedList.append(translated)


if __name__ == "__main__":
    en, es = separatePhrases('test.txt')
    testAccuracy(en, es)
