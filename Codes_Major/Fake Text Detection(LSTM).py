
################################# New Code (by LSTM) #################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

############# Load Data ---->
df = pd.read_csv('E:\\Users\\dell 6520\\Downloads\\Dataset_all_data.csv',delimiter=',',encoding='latin-1')
df.head()

############# Preprocessing ----->
df.info()
sns.countplot(df.type)
plt.xlabel('Label')
plt.title('Number of Real and Fake texts')

############### Create input and output vector ----->
X = df.text
Y = df.type
#le = LabelEncoder()
#Y = le.fit_transform(Y)
#Y = Y.reshape(-1,1)

################ Split test and train ----->
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20)
X_train = X_train.astype(str)
X_test = X_test.astype(str)
Y_train = Y_train.astype(str)
Y_test = Y_test.astype(str)

################ Process the data ------>
max_words = 1000
max_len = 300
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

labelencoder_Y = LabelEncoder()
Y_train = labelencoder_Y.fit_transform(Y_train)
Y_test = labelencoder_Y.fit_transform(Y_test)


################# LSTM Model ------>
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

################# Compile Model ------>
model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

################# Fitting the model ------->
model.fit(sequences_matrix,Y_train,batch_size=128,epochs=20,validation_split=0.2)

#model.fit(sequences_matrix,Y_train,batch_size=128,epochs=20,
#         validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

################## Evaluation on test set -------->
Y_pred = model.fit(test_sequences_matrix,Y_test,verbose = 1)

accr = model.evaluate(test_sequences_matrix,Y_test)

print('Test set  Loss: {:0.3f}  Accuracy: {:0.3f}'.format(accr[0],accr[1]))






