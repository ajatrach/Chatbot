
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from keras import models
from keras import layers
from keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


trainData=[]
validData=[]
testData=[]

#read the train data
trainFile = open("en.train.conll.txt","r",encoding="utf8",errors="ignore")
text=None
intent= None
for line in trainFile:
    line= line.strip()
    if line.startswith("# text:"):
        text=line.split("# text:")[1].strip()
    elif line.startswith("# intent:"):
        intent=line.split("# intent:")[1].strip()

    #make sure text and intent are not none
    if text is not None and intent is not None:
        trainData.append({"text": text, "intent": intent })
        text= None
        intent=None

#read the validation data
validFile = open("en.valid.conll.txt","r",encoding="utf8",errors="ignore")
text=None
intent= None
for line in validFile:
    line= line.strip()
    if line.startswith("# text:"):
        text=line.split("# text:")[1].strip()
    elif line.startswith("# intent:"):
        intent=line.split("# intent:")[1].strip()

    #make sure text and intent are not none
    if text is not None and intent is not None:
        validData.append({"text": text, "intent": intent })
        text= None
        intent=None

#read the test data
testFile = open("en.valid.conll.txt","r",encoding="utf8",errors="ignore")
text=None
intent= None
for line in testFile:
    line= line.strip()
    if line.startswith("# text:"):
        text=line.split("# text:")[1].strip()
    elif line.startswith("# intent:"):
        intent=line.split("# intent:")[1].strip()

    #make sure text and intent are not none
    if text is not None and intent is not None:
        testData.append({"text": text, "intent": intent })
        text= None
        intent=None

#sepparate sentences and intents
trainText, trainIntents, validText, validIntents, testText, testIntents =([] for i in range(6))
for item in trainData:
    trainText.append(item["text"])
    trainIntents.append(item["intent"])
for item in validData:
    validText.append(item["text"])
    validIntents.append(item["intent"])
for item in testData:
    testText.append(item["text"])
    testIntents.append(item["intent"])    

#find number of intents
labels=[]
for item in trainIntents:
    if item not in labels:
        labels.append(item)

vocabSize = 10000
maxSeqLen = 100
numIntents = len(labels)

#vectorize the labels
labelEncoder= LabelEncoder()
trainY = labelEncoder.fit_transform(trainIntents)
validY = labelEncoder.fit_transform(validIntents)
testY = labelEncoder.fit_transform(testIntents)

#tokenize train data
tokenizer= Tokenizer(num_words=vocabSize, oov_token = "<OOV>")
tokenizer.fit_on_texts(trainText)
trainSeq= tokenizer.texts_to_sequences(trainText)
trainX = padded_sequences = pad_sequences(trainSeq, maxlen=maxSeqLen, truncating='post')
#tokenize validation data
tokenizer.fit_on_texts(validText)
validSeq= tokenizer.texts_to_sequences(validText)
validX = padded_sequences = pad_sequences(validSeq, maxlen=maxSeqLen, truncating='post')
#tokenize test data
tokenizer.fit_on_texts(testText)
testSeq= tokenizer.texts_to_sequences(testText)
testX = padded_sequences = pad_sequences(testSeq, maxlen=maxSeqLen, truncating='post')



model= models.Sequential()
model.add(Embedding(input_dim=vocabSize, output_dim=numIntents, input_length=maxSeqLen))
model.add(GlobalAveragePooling1D())
#first
#model.add(Dense(128, activation='relu'))
#model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(2048, activation='relu'))


model.add(Dense(numIntents, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history=model.fit(trainX, trainY, epochs=5, batch_size=32, validation_split=0.2,validation_data=(validX, validY))
results=model.evaluate(testX, testY)
print(results)