import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.optimizers import SGD
import random

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:  #tokenizing each word in patterns
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #adding documents in the corpus
        documents.append((w, intent['tag']))

        #adding to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


#Lemmatizing, removing duplicates and converting each word to lowercase
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

#sorting classes list
classes = sorted(list(set(classes)))

#documents = combination between patterns and intents
print(len(documents),documents, "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

pickle.dump(words,open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#Creating our Training Data
training = []


output_empty = [0] * len(classes)

#training set, bag of words for each sentence
for doc in documents:
    #initializing bag of words
    bag =[]
    #list of tokenized words for the pattern
    pattern_words = doc[0]
    #lemmatizing each word
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)


    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])



random.shuffle(training)
training = np.array(training)


train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

#Creating a deep neural network consisting 3 layers.

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#Compiling model, stochastic gradient descent with Nesterov accelerated gradient gives good result for this model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print('Model Created')

