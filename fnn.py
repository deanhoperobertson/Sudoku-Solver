#feed-forward neural network for MNIST Dataset
import tensorflow as tf
import keras
from keras import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

def print_dataset_info():
	print("Training dataset:",X_train.shape[0])
	print("Test dataset:", X_test.shape[0])
	print("Clases:", len(set(y_train)))

	print(X_train[0])




