import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()
traineval = json.loads(open('TrainingData.JSON').read()) # read json
words = []
classes = []
belongings = []
ignore_letters = ['?', '!', '@', '.', ',']

for data in traineval['data']:
    for patterns in data['patterns']:
        word_list = nltk.word_tokenize(patterns) #lemmenizes the words under tag in the json
        words.extend(word_list) #appends the words to a wordlist
        belongings.append((word_list, data['tag']))
        if data['tag'] not in classes:
            classes.append(data['tag']) #checks for new tag data

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
pickle.dump(words, open('words.pkl', 'wb'))
print(words)

#ML section
train = []
output_0 = [0] * len(classes)
for belonging in belongings:
    case = []
    word_patterns = belonging[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
            case.append(1) if word in word_patterns else case.append(0)
    output_row = list (output_0)
    output_row[classes.index(belonging[1])] = 1
    train.append([case, output_row])
random.shuffle(train) #Shuffles the data
train = np.array(train)
train_x = list(train[:, 0])
train_y = list(train[:, 1]) #X and Y axis of the training data

print(case)