import numpy as np
from keras_preprocessing.sequence import pad_sequences
from tensorflow import keras
import train
import os
from gui import Ui_MainWindow
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QMainWindow, QApplication

model = keras.models.load_model(os.getcwd() + '\\finalReducedModel')
preEn, preEs, tkEn, tkEs, enInput = train.preprocess()


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # start local modifications to gui
        self.ui.translateButton.clicked.connect(self.translatePressed)

    def translatePressed(self):
        inputText = self.ui.englishBox.toPlainText().lower()
        print(inputText)

        # model = keras.models.load_model(os.getcwd() + '\\combined2model')
        # preEn, preEs, tkEn, tkEs, enInput = train.preprocess()

        # mirrors the key to value relationship in tkEn.word_index
        y_id_to_word = {value: key for key, value in tkEs.word_index.items()}

        print(y_id_to_word)
        y_id_to_word[0] = '<PAD>'

        sentence, original = inputText, inputText

        print(tkEn.word_index)
        # sentence equals the list of ids that correspond to each word in sentence
        sentence = [tkEn.word_index[word] for word in sentence.split()]
        print("[tkEn.word_index[word] for word in sentence.split()]", sentence)

        # post pad the sentence to the length of the n dimension
        sentence = pad_sequences([sentence], maxlen=preEn.shape[-1], padding='post')

        # print("preEn shape: {}".format(preEn.shape))
        # print("preEn shape[-1]: {}".format(preEn.shape[-1]))

        print("Sentence: ", sentence)
        print("Sentence shape: ", sentence.shape)

        sentences = np.array([sentence[0], preEn[0]])
        print(sentence)

        predictions = model.predict(sentences, len(sentences))

        print('Sample 1:')
        print(original)
        print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))

        print("\nPredictions: ", predictions[0])
        print(predictions.shape)
        print(predictions[0].shape)

        print('Sample 2:')
        print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[1]]))
        print(' '.join([y_id_to_word[np.max(x)] for x in preEs[0]]))

        outputText = ' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]
                               if y_id_to_word[np.argmax(x)] != "<PAD>"])
        self.ui.spanishBox.setText(outputText.capitalize())


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
