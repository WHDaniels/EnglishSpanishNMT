import random
from collections import OrderedDict
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import pickle
import numpy as np
from time import perf_counter


# combine both 'europarl' datasets into one
def combineEuroparl():
    with open('europarl-en.txt', 'r', encoding='utf-8') as fileEn, \
            open('europarl-es.txt', 'r', encoding='utf-8') as fileEs, \
            open('europarlCombined.txt', 'w', encoding='utf-8') as target:
        readEn = fileEn.readlines()
        readEs = fileEs.readlines()

        for x in range(len(readEn)):
            target.write(readEn[x].strip("\n") + "\t" + readEs[x])


# combine 'combined.txt' + 'Taboeba.txt' + 'europarlCombined.txt'
def combineThree():
    with open('combined.txt', 'r', encoding='utf-8') as file1, \
            open('Tatoeba.txt', 'r', encoding='utf-8') as file2, \
            open('europarlCombined.txt', 'r', encoding='utf-8') as file3, \
            open('reducedCombined.txt', 'w', encoding='utf-8') as target:

        read1, read2, read3 = file1.readlines(), file2.readlines(), file3.readlines()

        for x in range(len(read1)):
            target.write(read1[x])
        for x in range(len(read2)):
            target.write(read2[x])
        for x in range(len(read3)):
            target.write(read3[x])


# combine 'combined.txt' + 'Taboeba.txt'
def combineTwo():
    with open('combined2.txt', 'r', encoding='utf-8') as file1, \
            open('Tatoeba.txt', 'r', encoding='utf-8') as file2, \
            open('reducedCombined3 - Copy.txt', 'w', encoding='utf-8') as target:
        read1, read2 = file1.readlines(), file2.readlines()

        for x in range(len(read1)):
            line = read1[x].split("\t")
            if '\n' not in line[1]:
                target.write(line[0] + "\t" + line[1] + "\n")
            else:
                target.write(line[0] + "\t" + line[1])
        for x in range(len(read2)):
            line = read2[x].split("\t")
            if '\n' not in line[1]:
                target.write(line[0] + "\t" + line[1] + "\n")
            else:
                target.write(line[0] + "\t" + line[1])


# combine 'combined2.txt'
def combineOne():
    with open('combined2.txt', 'r', encoding='utf-8') as file1, \
            open('reducedCombined(no gov).txt', 'w', encoding='utf-8') as target:
        read1 = file1.readlines()

        for x in range(len(read1)):
            line = read1[x].split("\t")
            if '\n' not in line[1]:
                target.write(line[0] + "\t" + line[1] + "\n")
            else:
                target.write(line[0] + "\t" + line[1])


# shuffle 'reducedCombined.txt'
def shuffleReduced():
    with open('reducedCombined3', 'r', encoding='utf-8') as rFile, \
            open('reducedCombined2.txt', 'w', encoding='utf-8') as wFile:
        readf = rFile.readlines()
        random.shuffle(readf)

        for x in range(len(readf)):
            wFile.write(readf[x])


"""
Make multiple training data files based on if all words in the file are
in the top percentage of words in the full dataset
"""


# outfiles: (finalReduced2.txt, finalReduced3.txt, etc)

def topXInSet(outFile, x):
    with open('reducedCombined(no gov).txt', 'r', encoding='utf-8') as file, \
            open(outFile, 'w', encoding='utf-8') as target:
        f = file.readlines()

        random.shuffle(f)

        tk = Tokenizer()
        tk.fit_on_texts(f)

        tfList = []

        start = perf_counter()

        # x = 6 -> 9000 words
        # x = 8 -> 7000 words
        # x = 10 -> 5500 words
        # x = 15 -> 3500 words
        # x = 20 -> 2750 words
        stopNum = round(1 / x * (len(tk.word_index)))

        for n in range(len(f)):
            keep = True
            for word in f[n].split():
                if keep is True:
                    for num, entry in enumerate(list(tk.word_index.keys())):
                        word = ''.join(
                            c for c in word if c not in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n').lower()
                        if entry == word:
                            break
                        if num == stopNum:
                            keep = False
                            break
            tfList.append(keep)

        stop = perf_counter()

        i = 0
        for tf in tfList:
            if tf:
                i = i + 1

        print("Trues:", i)
        print("Time to finish:", stop - start)

        for x in range(len(f)):
            if tfList[x] is True:
                target.write(f[x])

        print("\n" + str(stopNum))
        print(len(f))
        print(len(tfList))


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
    with open("small_vocab_en.txt", "r", encoding='utf-8') as en, \
            open("small_vocab_fr.txt", "r", encoding='utf-8') as es, \
            open("new.txt", "w", encoding='utf-8') as new:
        enRead, esRead = en.readlines(), es.readlines()

        for j in range(len(enRead)):
            new.write(enRead[j].rstrip("\n").replace('.', '') + "\t" + esRead[j].replace('.', ''))


def loadData(translationFile):
    """
    Takes phrase data from a formatted txt file and loads it into two lists
    :param translationFile: file phrase data should be loaded from
    :return: two lists of phrases for english and spanish respectively
    """

    # make two lists to hold our english/spanish phrases
    enPhrases = []
    esPhrases = []

    # read phrase file and add each phrase to its respective list
    with open(translationFile, "r", encoding='utf-8') as file:
        content = file.readlines()

        print("Shuffling data")

        # shuffle the data before adding to list
        random.shuffle(content)
        print("Done Shuffling")

        for i in range(0, len(content[:])):
            line = content[i].split("\t")
            enPhrases.append(line[0])
            esPhrases.append(line[1])
            # print(line[0], " <---> ", line[1])

    return enPhrases, esPhrases


def firstWordOfFreqN(n, phraseList):
    """
    Gets the number of words to be allowed into the Tokenizer up until a specified cutoff
    :param n: cutoff number of words (a cutoff of 5 won't allow words with a frequency of 5 or lower to be tokenized)
    :param phraseList: list of phrases to be parsed
    :return: number of words before the cutoff (number of words that are allowed)
    """

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
    return 0


def preprocess(translationFile):
    """
    Preprocesses the phrase data into arrays that can be padded and trained on
    :return: two zero/post-padded arrays of integers, two english/spanish tokenizers
    """

    # get lists of both english and spanish phrases
    enPhrases, esPhrases = loadData(translationFile)

    """
    print("Test loadData")
    for x in range(len(enPhrases)):
        print(enPhrases[x], " <--> ", esPhrases[x])

    print("num entries: ", len(enPhrases), len(esPhrases))


    # word frequency cutoff
    n = 3

    englishNumWords = firstWordOfFreqN(n, enPhrases)
    spanishNumWords = firstWordOfFreqN(n, esPhrases)

    print("Actual english vocab: " + str(englishNumWords))
    print("Actual spanish vocab: " + str(spanishNumWords))

    # tokenizer setup
    tkEn = Tokenizer(num_words=englishNumWords + 2, oov_token="<UNK>", char_level=False,
                     filters='¡¿!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tkEs = Tokenizer(num_words=spanishNumWords + 2, oov_token="<UNK>", char_level=False,
                     filters='¡¿!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    """

    # tokenizer setup
    tkEn = Tokenizer()
    tkEs = Tokenizer()

    # create index based on word frequency
    tkEn.fit_on_texts(enPhrases)
    tkEs.fit_on_texts(esPhrases)

    print("word_count_en: {}".format(len(tkEn.word_counts.items())))
    print("word_count_es: {}\n".format(len(tkEs.word_counts.items())))

    print("tkEn.word_index: {}".format(len(tkEn.word_index)))
    print("tkEs.word_index: {}\n".format(len(tkEs.word_index)))

    # transform index contents to integers
    seqEn = tkEn.texts_to_sequences(enPhrases)
    seqEs = tkEs.texts_to_sequences(esPhrases)

    # post-pad english/spanish sequence to the length of the largest phrase in that list
    padEn = padding(seqEn)
    padEs = padding(seqEs)

    print("padEn.shape: {}".format(padEn.shape))
    print("padEs.shape: {}\n".format(padEs.shape))

    # reshape the second index to be 3-dimensional (for sparse categorical cross-entropy)
    padEs = padEs.reshape(*padEs.shape, 1)

    # amount of unique words in both languages
    print("Unique English words:", str(len(tkEn.word_index)))
    print("Unique Spanish words:", str(len(tkEs.word_index)), "\n")

    # pad and reshape the input so it can be used in the fit function
    # newEn = padding(seqEn, padEs.shape[1])
    newEn = padding(seqEn)
    # newEn = newEn.reshape((-1, padEs.shape[-2]))

    print("Preprocessing finished\n")
    pickle.dump(padEn, open('data//padEn.p', 'wb'))
    pickle.dump(padEs, open('data//padEs.p', 'wb'))
    pickle.dump(tkEn, open('data//tkEn.p', 'wb'))
    pickle.dump(tkEs, open('data//tkEs.p', 'wb'))


def padding(sequence, maxLength=None):
    """
    If the maxLength is not specified, get the length of the largest phrase through a list comprehension and use Keras'
    padding function to pad zeros to the end of each index up to the length of the largest phrase in that sequence
    :param sequence: array of integers that relate to words in the respective tokenizer's word_index
    :param maxLength: word length of largest phrase
    :return: a post-padded two dimensional array of integers
    """

    if maxLength is None:
        maxLength = max([len(phrase) for phrase in sequence])
    paddedSeq = pad_sequences(sequence, maxlen=maxLength, padding="post")

    return paddedSeq


def sequenceToText(sequence, tk):
    """
    Converts a sequence array back to word form
    :param sequence: array of integers that relate to words in the respective tokenizer's word_index
    :param tk: the respective tokenizer for the sequence array
    :return: a string of translated words
    """

    sequenceToWords = {wordID: word for word, wordID in tk.word_index.items()}
    sequenceToWords[0] = '<PAD>'

    return ' '.join([sequenceToWords[prediction] for prediction in np.argmax(sequence, 1)])


if __name__ == '__main__':
    # topXInSet("finalReduced2.txt", 6)
    preprocess("finalReduced.txt")
