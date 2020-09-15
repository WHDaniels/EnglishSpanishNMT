from keras_preprocessing.sequence import pad_sequences
from tensorflow import keras
import train
import os

model = keras.models.load_model(os.getcwd() + '\\model')


preEn, preEs, tkEn, tkEs, enInput = train.preprocess()

# mirrors the key to value relationship in tkEn.word_index
y_id_to_word = {value: key for key, value in tkEn.word_index.items()}
y_id_to_word[0] = '<PAD>'

sentence = 'he saw a old yellow truck'
# sentence equals the list of ids that correspond to each word in sentence
sentence = [tkEn.word_index[word] for word in sentence.split()]
# post pad the sentence to the length of the n dimension
"""
sentence = pad_sequences([sentence], maxlen=preEn.shape[-1], padding='post')
sentences = np.array([sentence[0], x[0]])
predictions = model.predict(sentences, len(sentences))

print('Sample 1:')
print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))
print('Il a vu un vieux camion jaune')
print('Sample 2:')
print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[1]]))
print(' '.join([y_id_to_word[np.max(x)] for x in y[0]]))
"""