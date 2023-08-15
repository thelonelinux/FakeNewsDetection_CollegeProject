################################# New Code (by LSTM) #################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from nltk.tokenize import sent_tokenize, word_tokenize 
import gensim 
from gensim.models import Word2Vec 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
#%matplotlib notebook

######################################### Load Data ###############################################

df = pd.read_csv('E:\\Users\\dell 6520\\Downloads\\Dataset_all_data.csv',delimiter=',',encoding='latin-1')
df.head()

####################################### Preprocessing ############################################

df.info()
sns.countplot(df.type)
plt.xlabel('Label')
plt.title('Number of Real and Fake texts')

################################# Create input and output vector ##################################

X = df[['text','title']]
Y = df.type
#le = LabelEncoder()
#Y = le.fit_transform(Y)
#Y = Y.reshape(-1,1)

###################################### Split test and train #######################################

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20)
X_train = X_train.astype(str)
X_test = X_test.astype(str)
Y_train = Y_train.astype(str)
Y_test = Y_test.astype(str)


############################# Segregation into "Text" and "Title" Part #############################

##### X1 = "Text" #######
X1_train = X_train.text
X2_train = X_train.title

X1_test = X_test.text
X2_test = X_test.text


################################# Label Encoding the Labels #######################################

labelencoder_Y = LabelEncoder()
Y_train = labelencoder_Y.fit_transform(Y_train)
Y_test = labelencoder_Y.fit_transform(Y_test)


######################################## Process the data #########################################

##############  1. Embeddings for "Text" ############

#max_words = 1000
#max_len = 300
#tok = Tokenizer(num_words=max_words)
#tok.fit_on_texts(X1_train)
#sequences_1 = tok.texts_to_sequences(X1_train)
#sequences_matrix_1 = sequence.pad_sequences(sequences_1,maxlen=max_len)

#test_sequences_1 = tok.texts_to_sequences(X1_test)
#test_sequences_matrix_1 = sequence.pad_sequences(test_sequences_1,maxlen=max_len)





















#############  2. Embeddings for "Title" ############

max_words = 1000
max_len = 300
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X2_train)
sequences_2 = tok.texts_to_sequences(X2_train)
sequences_matrix_2 = sequence.pad_sequences(sequences_2,maxlen=max_len)

test_sequences_2 = tok.texts_to_sequences(X2_test)
test_sequences_matrix_2 = sequence.pad_sequences(test_sequences_2,maxlen=max_len)



######################################### LSTM Model ###########################################

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=300)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

####################################### Compile Model ##########################################
    
model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])


##################################### Fitting the model #########################################

############# 1. Fitting Model on Text  ################

model.fit(sequences_matrix_1,Y_train,batch_size=128,epochs=20,validation_split=0.2)

############# 2. Fitting Model on Title ################

model.fit(sequences_matrix_2,Y_train,batch_size=128,epochs=50,validation_split=0.2)



######################################## Prediction ###############################################

############# 1. Predicting Text Part ###################

Y_pred_1 = model.predict(test_sequences_matrix_1)
Y_pred_1 = (Y_pred_1 > 0.5)

from sklearn.metrics import confusion_matrix
cm_1 = confusion_matrix(Y_test,Y_pred_1)

############# 1. Predicting Title Part ##################

Y_pred_2 = model.predict(test_sequences_matrix_2)
Y_pred_2 = (Y_pred_2 > 0.5)

from sklearn.metrics import confusion_matrix
cm_1 = confusion_matrix(Y_test,Y_pred_1)

#################################### Evaluation on test set ######################################

############# 1. Evaluation on Text #####################

accr_1 = model.evaluate(test_sequences_matrix_1,Y_test, verbose = 1)
print('Test set  Loss: {:0.3f}  Accuracy: {:0.3f}'.format(accr_1[0],accr_1[1]))

############# 2. Evaluation on Title ####################

accr_2 = model.evaluate(test_sequences_matrix_2,Y_test, verbose = 1)
print('Test set  Loss: {:0.3f}  Accuracy: {:0.3f}'.format(accr_2[0],accr_2[1]))


accr_1 = model.evaluate(sequences_matrix_1,Y_train, verbose = 1)








