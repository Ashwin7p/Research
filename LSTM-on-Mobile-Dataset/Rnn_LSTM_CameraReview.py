# Setting Environment Variable
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#importing all required libraries
#import json
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional




import pandas as pd
wk=pd.read_csv('Review_f.csv')
workbook=wk.iloc[:,[3,4]].values
df=pd.DataFrame(workbook)
siz=5000
df1=df[df[0]>=3]
df2=df[df[0]<3]
    
df1=df1[1]
df2=df2[1]
df1=df1[:siz]
df2=df2[:siz]

df1=pd.DataFrame(df1)
df2=pd.DataFrame(df2)
senti_review =[1]*len(df1)



# Adding the column of senti_review for df
df1['scores']=senti_review
senti_review=[]
senti_review =[0]*len(df2)
df2['scores']=senti_review
#df1 has only positive reviews and df2 has only negative reviews
# loading positive reviews to pos_data and negative reviews to neg_data
pos_data = df1
neg_data = df2

print("Posititve data loaded. ", len(pos_data), "entries")
print("Negative data loaded. ", len(neg_data), "entries")

print("done loading data...")

plabels = []
nlabels = []

# 2.Process reviews into sentences

# Storing positive sentences in pos_sentences
# Storing negative sentences in neg_sentences

pos_sentences, neg_sentences = [], [] 
for entry in pos_data[1] :
    pos_sentences.append(entry)
    plabels.append(1)
for entry in neg_data[1] :
    nlabels.append(0)
    neg_sentences.append(entry)
print(len(pos_sentences))
print(len(neg_sentences))

#texts contains positive review followed by negative review
texts = pos_sentences + neg_sentences 
# we first append 1 (as many positive reviews are there) and then we append 0(as many negative reviews(as many negative reviews are there))  
labels = [1]*len(pos_sentences) + [0]*len(neg_sentences)


print("after app", labels)
print(type(pos_sentences), len(pos_sentences), type(neg_sentences), len(neg_sentences))
print(type(texts), len(texts), type(labels), len(labels)) 


import re
for t in range(0,len(texts)):
    texts[t]=re.sub("^[a-zA-z]","",str(texts[t]))
# 3. Tokenize

# splitting sentences into tokens 
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
# Converting words to sequence
#each unique word is given different number(sequence number)
sequences = tokenizer.texts_to_sequences(texts)
#sequences is a list where in each sentences is converted to a list of indexes

#word_index is dictionary and each unique word in the review is given a index
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

MAX_SEQUENCE_LENGTH = 50
# we append zeros (padding is done) in the beginning of every element of list(sequences) so that all the sequences 
#..will be of same length that is equal to max_sequence_length(50 in this case) 
data = sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# print(labels)

# converting labels to an array of (n,1)
labels = np.array(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

#test train data split
from sklearn.cross_validation import train_test_split
x_train,x_val,y_train,y_val=train_test_split(data,labels,test_size=0.3,random_state=0)
print(len(x_train), len(y_train))

#GloVe

embeddings_index = {}
#in file f,each row begins with a word and then contains the vector representation of that word
f = open('glove.6B.50d.txt', 'r', encoding = 'utf-8')
for line in f:
    #extracting every line from file
    values = line.split()
    #word is first element of the values list
    word = values[0]
    #coefs is the vector representation of that word
    coefs = np.asarray(values[1:], dtype='float32')
    #embeddings_index is a dictionary where in key is the word and value is vector representation of that word
    embeddings_index[word] = coefs
    
f.close()

print('Found %s word vectors.' % len(embeddings_index))

EMBEDDING_DIM = MAX_SEQUENCE_LENGTH=50
# creating a zero matrix of shape(2004,50) in this case
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
# word_index contains our unique words from the tokenizer
for word, i in word_index.items():
    #embedding vector hs the vectors for the particular words 
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        # i is the number given the word by tokenizer
        #embedding vector rep of that word
        embedding_matrix[i] = embedding_vector

acc=[]
"""
Word embeddings provide a dense representation of words and their relative meanings.

They are an improvement over sparse representations used in simpler bag of word model representations.

Word embeddings can be learned from text data and reused among projects. They can also be learned as part of fitting a neural network on text data.
"""
from keras.layers import Embedding
seed = 7
np.random.seed(seed)
# Embedding layer is the first hidden layer

#input_dim: This is the size of the vocabulary in the text data. For example, our 
# data is integer encoded to values between 0-1984, then the size of the
# vocabulary would be 1985 words.

#output_dim: This is the size of the vector space in which words will be embedded.
# It defines the size of the output vectors from this layer for each word. For example,
# it is 50 in this case

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

def precision(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) 
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1))) 
    precision = true_positives / (predicted_positives + K.epsilon()) 
    return precision

def recall(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) 
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1))) 
    recall = true_positives / (possible_positives + K.epsilon()) 
    return recall 


# Training the LSTM model

batch_size = 128

model = Sequential()

# First hidden layer
model.add(embedding_layer)

model.add(LSTM(64))

model.add(Dropout(0.50))

#output layer
model.add(Dense(1, activation='sigmoid'))

# using adam optimizers 

model.compile('adam', 'binary_crossentropy', metrics=['accuracy', precision, recall])

print('Train...')

#fit the model to train set
history=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=100,
          validation_data=[x_val, y_val])
yhat=model.predict_classes(x_val)
#Plotting graph of accuracy vs Epoch number
plt.plot(np.arange(0,100),history.history['val_acc'])

#finding the final metrics such as loss, accuracy,precision,recall
x = model.evaluate(x_val,y_val)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_val,yhat)

print("Loss: ", x[0])
print("Accuracy: ", x[1])
print("Precision: ", x[2])
print("Recall: ", x[3])
