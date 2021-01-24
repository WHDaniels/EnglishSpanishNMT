import os
import _pickle as pickle

from keras import callbacks
from keras.layers import GRU, Dense, TimeDistributed, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from tensorflow.python.keras.models import Sequential


if __name__ == '__main__':
    # combineData()
    # combined = open('combined.txt', 'r', encoding='utf-8')

    padEn, padEs = pickle.load(open('data//padEn.p', 'rb')), pickle.load(open('data//padEs.p', 'rb'))
    tkEn, tkEs = pickle.load(open('data//tkEn.p', 'rb')), pickle.load(open('data//tkEs.p', 'rb'))

    print("English Vocab: {}".format(len(tkEn.word_index)))
    print("\nSpanish Vocab: {}".format(len(tkEs.word_index)))

    # try 5e-4, .0005
    learningRate = 0.001
    model = Sequential()

    # add layers
    model.add(Embedding(input_dim=len(tkEn.word_index) + 1, output_dim=128, input_length=padEn.shape[1]))
    model.add(Bidirectional(GRU(256, return_sequences=False)))
    model.add(RepeatVector(padEs.shape[1]))
    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add(TimeDistributed(Dense(len(tkEs.word_index) + 1, activation="softmax")))

    # compiles model
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learningRate),
                  metrics=['accuracy'])

    # callback to save best model
    checkpointPath = (os.getcwd() + '\\FINAL1')
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpointPath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # trains model
    model.fit(padEn, padEs, batch_size=64, epochs=50, validation_split=0.2, callbacks=[model_checkpoint_callback])

    # saves model after training
    # savePath = 'C:\\Users\\mercm\\OneDrive\\Documents\\GitHub\\EnglishSpanishNMT\\model'
    # save_model(model, savePath)

    # print(sequenceToText(model.predict(enInput[:1])[0], tkEs))
