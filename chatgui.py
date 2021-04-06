import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

# import keras
from keras.models import load_model
#load model
model = load_model('chatbot_model.h5')
# import json for intents
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


import nltk
nltk.download('punkt')
nltk.download('wordnet')
# for cleaning up the text
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
   
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(text, words, show_details=True):
    # words are shortened
    sent_words = clean_up_sentence(text)
    # matrix vocabulary words
    bag = [0]*len(words)  
    for s in sent_words:
        for i,w in enumerate(words):
            if w == s: 
                # if value is present bag=1
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    # 0 or 1 if te value is in sentences
    return(np.array(bag))

# filters below a 0.25 
def predict_text(text, model):
    # filter or remove predictions which are below a threshold
    t = bow(text, words,show_details=False)
    p = model.predict(np.array([t]))[0]
    # threshold value is 0.25
    ERROR_THRESHOLD = 0.25
    # the results are filter above 0.25
    results = [[i,r] for i,r in enumerate(p) if r>ERROR_THRESHOLD]
    # order using n probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_words = []
    for r in results:
        return_words.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_words

# get the text
def getResponse(ints, intents_json):
    label = ints[0]['intent']
    # intent has array
    intent_list = intents_json['intents']
    for i in intent_list:
        if(i['tag']== label):
            cal = random.choice(i['responses'])
            break
    # return text
    return cal

# for the chatbot message
def response_chatbot(message):
    in1 = predict_text(message, model)
    response_msg = getResponse(in1, intents)
    return response_msg


#Creating GUI with tkinter
import tkinter
from tkinter import *


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
    
        res = response_chatbot(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
            
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
 

base = Tk()
# the name
base.title("TEXT MODEL")
# provide coordinates
base.geometry("400x500")
# if chat can be maximize
base.resizable(width=FALSE, height=FALSE)

ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)
# scroll bar
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )
# for input text
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
# place the scroll bar x , y  and height
scrollbar.place(x=376,y=6, height=386)
# chatlog x y height
ChatLog.place(x=6,y=6, height=386, width=370)
# entry box
EntryBox.place(x=128, y=401, height=90, width=265)
# Send text
SendButton.place(x=6, y=401, height=90)

base.mainloop()
