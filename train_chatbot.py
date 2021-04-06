#import the packages
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import pandas as pd

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

import nltk
nltk.download('punkt')
nltk.download('wordnet')

# create a array for required
words=[]
list_words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

#the intents is iterated for each value 
for intent in intents['intents']:
    #pattern is iterated
    for pattern in intent['patterns']:
        #the pattern is split into words
        y = nltk.word_tokenize(pattern)
        words.extend(y)
        documents.append((y, intent['tag']))

        # If the value is present it is not added classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

for intent in intents['intents']:
    count+=1



# remove existing values
words = [lemmatizer.lemmatize(y.lower()) for y in words if y not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
# get length of classes
print (len(classes), "classes", classes)


# get classes
#get the number of unique
print (len(documents), "documents")

#get unique
print (len(words), "unique words", words)


# change the words to file
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# training
# array for train
training = []
# output empty
temp = [0] * len(classes)
# check documents
for doc in documents:
    # list of words
    gword = []
    # tokenized words
    z = doc[0]
    z = [lemmatizer.lemmatize(word.lower()) for word in z]
    #  the word is found in pattern add list words 
    for w in words:
        gword.append(1) if w in z else gword.append(0)
    
    # output is either 1 or 0
    z_row = list(temp)
    z_row[classes.index(doc[1])] = 1
    training.append([bag, z_row])
# shuffle
random.shuffle(training)
# change to np
training = np.array(training)
# training and test lists, X are patterns, Y are intents
training_x = list(training[:,0])
training_y = list(training[:,1])
print("Training data")


# Create model. 1st contains 128 neurons,second contains 64 neurons and 3rd output layer contains number of intents neurons
# the model is used to get output
# accuracy is used
model = Sequential()
# input layers dense, activation,
model.add(Dense(128, input_shape=(len(training_x[0]),), activation='relu'))
# the input layer with 0.5
model.add(Dropout(0.5))
# second layer
model.add(Dense(64, activation='relu'))
# next layer
model.add(Dropout(0.5))
# type of activation is  relu 
# for output softmax
model.add(Dense(len(training_y[0]), activation='softmax'))

# Stochastic gradient descent with Nesterov accelerated gradient
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# compile model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
k = model.fit(np.array(training_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', k)

print("model created")

# accuracy is used
n1 = Sequential()
# input layers dense, activation,
n1.add(Dense(64, input_shape=(len(training_x[0]),), activation='relu'))
# the input layer with 0.5
n1.add(Dropout(0.5))
# second layer
n1.add(Dense(32, activation='relu'))
# next layer
n1.add(Dropout(0.5))
# type of activation is  relu 
# for output softmax
n1.add(Dense(len(training_y[0]), activation='softmax'))

# Stochastic gradient descent with Nesterov accelerated gradient
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# compile model
n1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
k = n1.fit(np.array(training_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
n1.save('chatbot_model9.h5', k)

# accuracy is used
n2 = Sequential()
# input layers dense, activation,
n2.add(Dense(32, input_shape=(len(training_x[0]),), activation='relu'))
# the input layer with 0.5
n2.add(Dropout(0.5))
# second layer
n2.add(Dense(64, activation='relu'))
# next layer
n2.add(Dropout(0.5))
# type of activation is  relu 
# for output softmax
n2.add(Dense(len(training_y[0]), activation='softmax'))

# Stochastic gradient descent with Nesterov accelerated gradient
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# compile model
n2.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
k = n2.fit(np.array(training_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
n2.save('chatbot_model99.h5', k)