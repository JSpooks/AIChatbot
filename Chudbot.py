import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
traineval = json.loads(open('TrainingData.JSON').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('LoopSpooksmodel.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def wordbag(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predictclass(sentence): #prediction function
    wb = wordbag(sentence)
    res = model.predict(np.array([wb]))[0]
    Error_Threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > Error_Threshold]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'datum': classes[r[0]], 'probability': str(r[1])})
        return return_list

def get_response(data_list, traineval_json): #Response to pattern
    tag = data_list[0]['datum']
    list_of_data = traineval_json['data']
    for i in list_of_data:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("Bot running")


while True:
    message = input("").lower()
    ints = predictclass(message)
    res = get_response(ints, traineval)
    print(res)
