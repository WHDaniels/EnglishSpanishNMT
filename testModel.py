import pickle
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from tensorflow import keras
import os
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu

padEn, padEs = pickle.load(open('data//padEn.p', 'rb')), pickle.load(open('data//padEs.p', 'rb'))
tkEn, tkEs = pickle.load(open('data//tkEn.p', 'rb')), pickle.load(open('data//tkEs.p', 'rb'))


def separatePhrases(testFile):
    enPhrases, esPhrases = list(), list()

    with open(testFile, "r", encoding="utf-8") as test:
        testRead = test.readlines()

        for n, pairing in enumerate(testRead):
            separate = pairing.split("\t")
            enPhrases.append(''.join(c for c in separate[0]
                                     if c not in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n').lower())
            esPhrases.append(''.join(c for c in separate[1]
                                     if c not in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n').lower())

    return enPhrases, esPhrases


def testAccuracy(enPhrases, esPhrases, modelName):
    model = keras.models.load_model(os.getcwd() + '\\' + modelName)

    translatedList = list()
    candidateList = list()
    referenceList = list()

    skipCounter = 0
    for n, phrase in enumerate(enPhrases):
        # mirrors the key to value relationship in tkEn.word_index
        y_id_to_word = {value: key for key, value in tkEs.word_index.items()}
        y_id_to_word[0] = '<PAD>'

        original = enPhrases[n]
        sentence = enPhrases[n]
        # print(sentence)

        skip = skipCounter
        for word in sentence.split():
            if word not in tkEn.word_index:
                skipCounter += 1
                break

        if skipCounter > skip:
            continue

        # sentence equals the list of ids that correspond to each word in sentence
        sentence = [tkEn.word_index[word] for word in sentence.split()]

        # post pad the sentence to the length of the n dimension
        sentence = pad_sequences([sentence], maxlen=padEn.shape[-1], padding='post')
        sentences = np.array([sentence[0], padEn[0]])

        predictions = model.predict(sentences, len(sentences))

        translated = ' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]
                               if y_id_to_word[np.argmax(x)] != "<PAD>"])

        translatedList.append(translated)

        print("Translated:", translated)
        print("esPhrases[n]:", esPhrases[n], "\n")
        candidate = [token for token in translated.split()]
        reference = [token for token in esPhrases[n].split()]

        candidateList.append(candidate)
        referenceList.append([reference])

    score = corpus_bleu(referenceList, candidateList, weights=(1, 0, 0, 0))
    print("\nThe average BLEU score for this model is ", score)


if __name__ == "__main__":
    en, es = separatePhrases('test.txt')
    testAccuracy(en, es, "FINAL4")
