import os
import _pickle as pickle

from sklearn.model_selection import train_test_split
from keras import callbacks
from keras.layers import GRU, Dense, TimeDistributed, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.models import load_model
from tensorflow.python.keras.models import Sequential

if __name__ == '__main__':
    # combineData()
    # combined = open('combined.txt', 'r', encoding='utf-8')

    padEn, padEs = pickle.load(open('data//padEn.p', 'rb')), pickle.load(open('data//padEs.p', 'rb'))
    tkEn, tkEs = pickle.load(open('data//tkEn.p', 'rb')), pickle.load(open('data//tkEs.p', 'rb'))

    print("English Vocab: {}".format(len(tkEn.word_index)))
    print("\nSpanish Vocab: {}".format(len(tkEs.word_index)))

    # try 5e-4, .0005
    # learningRate = 0.001
    learningRate = 0.0001

    # Remove comments below when loading previously trained model for continued training
    # """
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
    # """

    # Add comments below to load previously trained model (MAKE SURE TO CHANGE SAVE PATH)
    # Load previously trained model
    """
    model = load_model('FINAL2')
    model.summary()

    loss, acc = model.evaluate(padEn, padEs)
    print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
    """

    # callback to save best model
    checkpointPath = (os.getcwd() + '\\FINAL4')
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpointPath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # trains model
    model.fit(padEn, padEs, batch_size=256, epochs=300, validation_split=0.2, callbacks=[model_checkpoint_callback])
