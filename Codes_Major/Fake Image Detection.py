import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report

############## Part 1 - Initialising the CNN ------>
classifier = Sequential()

###### Convolution ->
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

###### Pooling ->
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

###### Flattening ->
classifier.add(Flatten())

###### Full Connection ->
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

###### Compilation ->
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

############### Part 2 - Fitting the CNN to the images ------->
from keras.preprocessing.image import ImageDataGenerator

###### 1. Prepare Datagen object for training and test set ---->
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

###### 2. Use the Datagen object ----->
training_set = train_datagen.flow_from_directory('Fake News Dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 4,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('Fake News Dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 4,
                                            class_mode = 'binary')

################# Part -3 evaluation by cross validation ------>
classifier.fit_generator(training_set,
                         steps_per_epoch = 153,
                         epochs = 15,
                         validation_data = test_set,
                         validation_steps = 48,verbose =0)

################# Part -4 Prediction ---------->

#Y_pred = classifier.predict_generator(test_set, 48 // 5)
#Y_pred = np.argmax(Y_pred, axis=1)
#cm_1 = confusion_matrix(test_set,Y_pred)
#print('Confusion Matrix')
#print(confusion_matrix(test_datagen, Y_pred))
#print('Classification Report')
#target_names = ['Real','Fake']
#print(classification_report(test_datagen.classes, Y_pred, target_names=target_names))


