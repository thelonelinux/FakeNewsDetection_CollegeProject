import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import urllib
import os
import urllib.request
from urllib.request import urlopen, Request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model, Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Conv2D, MaxPooling2D, Flatten
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
X = df.main_img_url
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
N1=len(X_train.index)
N2=len(X_test.index)

for items in X_train.iteritems(): 
    if(Y_train[items[0]]=="real"):
        os.chdir(r"E:\Users\dell 6520\Fake News Dataset\X_train_img\real")
        urllib.request.urlretrieve(items[1],"real"+str(items[0])+".jpg")
        os.chdir("../")
    else:
        os.chdir(r"E:\Users\dell 6520\Fake News Dataset\X_train_img\fake")
        urllib.request.urlretrieve(items[1], "fake"+str(items[0])+".jpg")
        os.chdir("../")
        
        
        
        
url = 'https://t4.rbxcdn.com/c5695e5f087535e2066dc473e03b1819'
urllib.request.urlretrieve(url,"real"+".jpg")

count=0
countr=0
countT=0
for items in X_train.iteritems(): 
    countT+=1
    if(Y_train[items[0]]=="real"):
        countr+=1
        if(items[1][8]=='t' and items[1][7] is not 's'):
            print(items[0],items[1])
            count+=1
    
print (countT, countr, count)

print(resp.read())



        
for items in X_train.iteritems(): 
    print(items[0])

labelencoder_Y = LabelEncoder()
Y_train = labelencoder_Y.fit_transform(Y_train)
Y_test = labelencoder_Y.fit_transform(Y_test)


################# LSTM Model ------>
def CNN():
    layer = Sequential()
    layer.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    layer.add(MaxPooling2D(pool_size = (2, 2)))
    layer.add(Conv2D(32, (3, 3), activation = 'relu'))
    layer.add(MaxPooling2D(pool_size = (2, 2)))
    layer.add(Flatten())
    layer.add(Dense(units=128,name='FC1',activation='relu'))
    layer.add(Dropout(0.5))
    layer.add(Dense(units=1,name='out_layer',activation='sigmoid'))
    return layer

################# Compile Model ------>
model = CNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

################# Fitting the model ------->
model.fit(X_train,Y_train,batch_size=128,epochs=20,validation_split=0.2)

#model.fit(sequences_matrix,Y_train,batch_size=128,epochs=20,
#         validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

################## Evaluation on test set -------->
Y_pred = model.fit(X_test,Y_test,verbose = 1)

accr = model.evaluate(X_test,Y_test)

print('Test set  Loss: {:0.3f}  Accuracy: {:0.3f}'.format(accr[0],accr[1]))






