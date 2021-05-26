#feed-forward neural network for MNIST Dataset
import pandas as pd
import tensorflow as tf
import keras
from keras import Sequential
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from sklearn.metrics import confusion_matrix


EPOCHS = 5
BATCH = 128

#load MNIST dataset from keras
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

def remove(digit, x , y):
	idx = (y != digit).nonzero()
	return x[idx], y[idx]

X_train, Y_train = remove(0, X_train, Y_train)
X_test, Y_test = remove(0, X_test, Y_test)

#cast type
X_train = X_train.reshape(X_train.shape[0], 1,28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
Y_train = Y_train.astype('int32')
Y_test = Y_test.astype('int32')

#rescale data
X_train = X_train/255
X_test = X_test/255

#one-hot encode the target variables
y_train = np_utils.to_categorical(Y_train)
y_test = np_utils.to_categorical(Y_test)

#Build neural network architecture
#--------MODEL 1 ---------------
model = Sequential()
model.add(Conv2D(10, (5, 5), input_shape=(1, 28, 28), activation='relu', data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

#--------MODEL 2 ---------------
# model = Sequential()
# model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu', data_format='channels_first'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(15, (3, 3), activation='relu')) ## second convolutiuonal layer
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(10, activation='softmax'))

#compile model
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())

#fit model to training data
model.fit(X_train,y_train,
          validation_split=0.2,
          verbose=1,
          batch_size=BATCH, 
          epochs=EPOCHS)

#evaludate performance
final_loss, final_acc = model.evaluate(X_test,y_test,verbose=0)
print("Test loss: {0:.2f}, Test accuracy: {1:.2f}%".format(final_loss, final_acc*100))


#fetch test predictions
predictions = model.predict_classes(X_test)
preds=predictions.tolist()
y_lbls = Y_test.tolist()

#display confusion matrix
cm_1 = confusion_matrix(y_lbls, preds)
print(pd.DataFrame(cm_1))


#now to save weightings of trained model
model_json = model.to_json()
with open("models/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/model.h5")
print("Saved model to disk")
