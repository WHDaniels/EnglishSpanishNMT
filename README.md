# EnglishSpanishNMT

## Summary
This is a program made to translate Spanish phrases to their English counterpart through a trained model.
I take a Spanish-English parallel corpus and preprocess this into data which I can feed into a multi-layer
recurrent neural network model to get accurate predictions. PyQt5 is used as the user interface where input can be
given to the model.

## Data taken
[Tatoeba corpus](http://opus.nlpl.eu/Tatoeba.php)
[Article here](http://www.lrec-conf.org/proceedings/lrec2012/pdf/463_Paper.pdf)

## Data Preprocessing
Through parsing the entire Tatoeba dataset, a file is created that contains only the phrase pairings that contain 
the top 20% most used words in the dataset. This allows for less noise in the data, although restricting the
variety of words the model sees in training. The percentage can be changed to allow for more or less word inclusion.

Phrases in this file are shuffled to enable the test and training sets to contain the same lengths of sentences on average. 
The words in the phrases in this file are separated by language, then tokenized and put into integer format based upon their frequency through a Keras Tokenizer. Instead of a list of phrases, there are now two two-dimensional arrays of integers 
which can be translated to their respective words at any time. These two arrays are then padded to the length of their
longest phrase.

The padded English array is our model input, and the padded Spanish array is the model output.

## Model Training
A Keras Sequential model with five layers (Embedding, two Bidirectional, RepeatVector, and a Dense layer with a 
softmax activation) is used. This model is trained for 50 epochs with a batch size of 64, learning rate of .005, 
and takes a 20 percent validation split.

## User Interface
Using the PyQt5 library, a MainWindow class holds two text boxes which takes English input and outputs the Spanish
prediction of the model.