import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')

import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    #tokenizing the pattern - splitting the words into array
    sentence_words = nltk.word_tokenize(sentence)
    #stemming each word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details = True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return (np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words, show_details = False)
    res = model.predict(np.array([p]))[0]
    error_threshhold = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > error_threshhold]
    results.sort(key = lambda x:x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res


#Implementing GUI

import tkinter
from tkinter import *


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#FDD428", font=("Helvetica", 12, 'bold' ))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Codeclause Chatbot:  " + res + '\n\n\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("Codeclause Chatbot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)



ChatLog = Text(base, bd=0, bg="#002E5B",height=10, width=100, font="Arial",)
ChatLog.config(state=DISABLED)


scrollbar = Scrollbar(base, command=ChatLog.yview)
ChatLog['yscrollcommand'] = scrollbar.set


SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width=10, height=3,
                    bd=1, bg="#FDD428", activebackground="#ececec",fg='#000000',
                    command= send )


EntryBox = Text(base, bd=2, bg="white",width=29, height=3, font="Arial")
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(relheight=0.100,relwidth=0.77,relx=0.01,rely=0.80)
SendButton.place(relx=0.79,rely=0.80,relheight=0.100,relwidth=0.20)

base.mainloop()

