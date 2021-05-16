#feed-forward neural network for MNIST Dataset
import tensorflow as tf
import keras
from keras import Sequential
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten


EPOCHS = 5
BATCH = 128

#load MNIST dataset from keras
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#cast type
X_train = X_train.reshape(X_train.shape[0], 1,28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
y_train = y_train.astype('int32')
y_test = y_test.astype('int32')

#rescale data
X_train = X_train/255
X_test = X_test/255

#one-hot encode the target variables
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


def print_dataset_info():
	print("Training input:",X_train.shape)
	print("Training output:", y_train.shape)


model = Sequential()
model.add(Conv2D(10, (5, 5), input_shape=(1, 28, 28), activation='relu', data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())


# model.fit(X_train,y_train,
#           validation_split=0.2,
#           verbose=1,
#           batch_size=BATCH, 
#           epochs=EPOCHS)



