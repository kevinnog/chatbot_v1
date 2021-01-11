import random
import json
import pickle
import numpy as np
import nltk
import PySimpleGUI as sg
from nltk.stem import  WordNetLemmatizer
from tensorflow.keras.models import load_model

import sys
print("PATH", sys.path)


sg.theme('dark green 5')

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('app/intents.json').read())

words = pickle.load(open('app/words.pkl', 'rb'))
classes = pickle.load(open('app/classes.pkl', 'rb'))

model = load_model('app/chatbotmodel.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    
    for r in results:
        return_list.append({'intents': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intent_list, intents_json):
    tag = intent_list[0]['intents']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result



# Define the window's contents
layout = [[sg.Text("Hi friend, how can i help you?")],
          [sg.Input(size=(100,1), key='-INPUT-')],
          [sg.Text(size=(100,45), key='-OUTPUT-')],
          [sg.Button('Ok'), sg.Button('Quit')]]

# Create the window
window = sg.Window('Tobias 9000', layout, location=(0,0), size=(800,600), keep_on_top=True)

# Display and interact with the Window using an Event Loop
while True:
    event, values = window.read()
    # See if user wants to quit or window was closed
    if event == sg.WINDOW_CLOSED or event == 'Quit':
        break
    # Output a message to the window
    ints = predict_class(values['-INPUT-'])
    res = get_response(ints, intents)
    window['-OUTPUT-'].update(res)

# Finish up by removing from the screen
window.close()

# while True:
#     message = input("")
#     ints = predict_class(message)
#     res = get_response(ints, intents)
#     print(res)
