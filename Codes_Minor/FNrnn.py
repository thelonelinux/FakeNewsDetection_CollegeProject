from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Fake News Dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 4,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('Fake News Dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 4,
                                            class_mode = 'binary')

# Part 1 - Initialising the CNN
classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

# Adding the first LSTM layer and some Dropout regularisation
classifier.add(LSTM(units = 4,return_sequences = True, input_shape = (training_set.shape[1], 1))) #//////////////
classifier.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
classifier.add(LSTM(units = 4, return_sequences = True))
classifier.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
#classifier.add(LSTM(units = 4, return_sequences = True))
#classifier.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
classifier.add(LSTM(units = 4))
classifier.add(Dropout(0.2))

# Adding the output layer
classifier.add(Dense(units = 1))

classifier.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Part 2 - Fitting the CNN to the images
classifier.fit_generator(training_set,
                         steps_per_epoch = 153,
                         epochs = 15,
                         validation_data = test_set,
                         validation_steps = 48)