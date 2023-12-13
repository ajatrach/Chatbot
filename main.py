
import json
import numpy as np
import tensorflow as tf
import kerastuner as kt
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
from keras.layers import Dropout


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

#separate sentences and intents
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
numIntents = len(labels)
#constants for tokenizer/model
vocabSize = 10000
maxSeqLen = 100

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


#Dummy classifier for comparison
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

dummy = DummyClassifier(strategy="stratified")
dummy.fit(trainX, trainY)
dummyPred = dummy.predict(testX)
dummyAcc = accuracy_score(testY, dummyPred)
print("Dummy model accuracy: ", dummyAcc)


#making a hypermodel for tuning
def modelBuilder(hp):
    model= models.Sequential()
    model.add(Embedding(input_dim=vocabSize, output_dim=numIntents, input_length=maxSeqLen))
    model.add(GlobalAveragePooling1D())

    #find optimal number of units in the first layer
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
   
    model.add(Dense(units=hp_units, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16))
    model.add(Dense(16))
    model.add(Dense(16))
    model.add(Dense(16))
    model.add(Dense(numIntents, activation='softmax'))

    #tune learning rate from 0.01, 0.001, or 0.0001
    hpLearningRate = hp.Choice('learning_rate', values=[0.01, 0.001, 0.0001])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hpLearningRate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

#tune hyperparameters
tuner = kt.Hyperband(
    modelBuilder,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='Chatbot',
    project_name='tuning')

#Find the best hyperparameters
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(trainX, trainY, epochs=50, validation_data=(validX, validY), callbacks=[stop_early])

# Get the optimal hyperparameters
bestHyp=tuner.get_best_hyperparameters(num_trials=1)[0]
print(bestHyp.values)
print("Best number of units for first dense layer: ", bestHyp.get('units'), "best learning rate optimizer: ",bestHyp.get('learning_rate'))

#Find best number of epochs through training
model = tuner.hypermodel.build(bestHyp)
history = model.fit(trainX, trainY, epochs=25, validation_data=(validX, validY))

accuracyPerEpoch = history.history['val_accuracy']
bestEpoch = accuracyPerEpoch.index(max(accuracyPerEpoch)) + 1
print('Best epoch: ',bestEpoch)

#now retrain with best epoch
hypermodel = tuner.hypermodel.build(bestHyp)
hypermodel.fit(trainX, trainY, epochs=bestEpoch, validation_data=(validX, validY))

result = hypermodel.evaluate(testX, testY)
print(result)

