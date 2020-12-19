from random import shuffle

# combine 'europarl'
"""
with open('europarl-en.txt', 'r', encoding='utf-8') as fileEn, \
        open('europarl-es.txt', 'r', encoding='utf-8') as fileEs, \
        open('europarlCombined.txt', 'w', encoding='utf-8') as target:
    readEn = fileEn.readlines()
    readEs = fileEs.readlines()

    for x in range(len(readEn)):
        target.write(readEn[x].strip("\n") + "\t" + readEs[x])
"""

"""
# combine 'combined.txt' + 'Taboeba.txt' + 'europarlCombined.txt'

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
"""

# or


# combine 'combined.txt' + 'Taboeba.txt'
"""
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
"""

# combine 'combined2.txt'
"""
with open('combined2.txt', 'r', encoding='utf-8') as file1, \
        open('reducedCombined(no gov).txt', 'w', encoding='utf-8') as target:
    read1 = file1.readlines()

    for x in range(len(read1)):
        line = read1[x].split("\t")
        if '\n' not in line[1]:
            target.write(line[0] + "\t" + line[1] + "\n")
        else:
            target.write(line[0] + "\t" + line[1])
"""

# shuffle 'reducedCombined.txt'
"""
with open('reducedCombined3', 'r', encoding='utf-8') as rFile, \
        open('reducedCombined2.txt', 'w', encoding='utf-8') as wFile:
    readf = rFile.readlines()
    shuffle(readf)

    for x in range(len(readf)):
        wFile.write(readf[x])
"""
# look up most used english words --> spanish translations
# or
# get top percentage of both english and spanish words


from keras_preprocessing.text import Tokenizer
from time import perf_counter

with open('reducedCombined(no gov).txt', 'r', encoding='utf-8') as file, \
        open('finalReduced2.txt', 'w', encoding='utf-8') as target:
    f = file.readlines()

    shuffle(f)

    tk = Tokenizer()
    tk.fit_on_texts(f)

    tfList = []

    start = perf_counter()

    # around 9000
    stopNum = round(1 / 6 * (len(tk.word_index)))

    for n in range(len(f)):
        keep = True
        for word in f[n].split():
            if keep is True:
                for num, entry in enumerate(list(tk.word_index.keys())):
                    word = ''.join(c for c in word if c not in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n').lower()
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

    with open('reducedCombined(no gov).txt', 'r', encoding='utf-8') as file, \
            open('finalReduced3.txt', 'w', encoding='utf-8') as target:
        f = file.readlines()

        shuffle(f)

        tk = Tokenizer()
        tk.fit_on_texts(f)

        tfList = []

        start = perf_counter()

        # around 7000
        stopNum = round(1 / 8 * (len(tk.word_index)))

        for n in range(len(f)):
            keep = True
            for word in f[n].split():
                if keep is True:
                    for num, entry in enumerate(list(tk.word_index.keys())):
                        word = ''.join(c for c in word if c not in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n').lower()
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

        with open('reducedCombined(no gov).txt', 'r', encoding='utf-8') as file, \
                open('finalReduced4.txt', 'w', encoding='utf-8') as target:
            f = file.readlines()

            shuffle(f)

            tk = Tokenizer()
            tk.fit_on_texts(f)

            tfList = []

            start = perf_counter()

            # around 5500
            stopNum = round(1 / 10 * (len(tk.word_index)))

            for n in range(len(f)):
                keep = True
                for word in f[n].split():
                    if keep is True:
                        for num, entry in enumerate(list(tk.word_index.keys())):
                            word = ''.join(c for c in word if c not in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n').lower()
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

            with open('reducedCombined(no gov).txt', 'r', encoding='utf-8') as file, \
                    open('finalReduced5.txt', 'w', encoding='utf-8') as target:
                f = file.readlines()

                shuffle(f)

                tk = Tokenizer()
                tk.fit_on_texts(f)

                tfList = []

                start = perf_counter()

                # around 3500
                stopNum = round(1 / 15 * (len(tk.word_index)))

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

                with open('reducedCombined(no gov).txt', 'r', encoding='utf-8') as file, \
                        open('finalReduced6.txt', 'w', encoding='utf-8') as target:
                    f = file.readlines()

                    shuffle(f)

                    tk = Tokenizer()
                    tk.fit_on_texts(f)

                    tfList = []

                    start = perf_counter()

                    # around 2750
                    stopNum = round(1 / 20 * (len(tk.word_index)))

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