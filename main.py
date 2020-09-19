import numpy as np
from keras_preprocessing.sequence import pad_sequences
from tensorflow import keras
import train
import os

model = keras.models.load_model(os.getcwd() + '\\europarl_model')


preEn, preEs, tkEn, tkEs, enInput = train.preprocess()

# mirrors the key to value relationship in tkEn.word_index
y_id_to_word = {value: key for key, value in tkEs.word_index.items()}
y_id_to_word[0] = '<PAD>'

print("Input: ")
sentence = input()
original = sentence

# sentence equals the list of ids that correspond to each word in sentence
sentence = [tkEn.word_index[word] for word in sentence.split()]
print(sentence)
# post pad the sentence to the length of the n dimension
sentence = pad_sequences([sentence], maxlen=preEn.shape[-1], padding='post')
print("preEn shape: {}".format(preEn.shape))
print("preEn shape[-1]: {}".format(preEn.shape[-1]))

print("Sentence: {}".format(sentence))
print("Sentence shape: {}".format(sentence.shape))

sentences = np.array([sentence[0], preEn[0]])
print(sentence)

predictions = model.predict(sentences, len(sentences))

print('Sample 1:')
print(original)
print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))

print("\nPredictions: {}".format(predictions[0]))

print('Sample 2:')
print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[1]]))
print(' '.join([y_id_to_word[np.max(x)] for x in preEs[0]]))
