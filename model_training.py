"""
Script that trains the multi-layer perceptron.
Will print the accuracy on the unseen, test data.
Can be trained either with the normal or the extended database (Note: the extended database takes significantly more time to train)
Comment the corresponding lines to chose the training data. Right now it uses the extended database.
"""

import minst_loader
import DigitsRecognizingNN
import gzip
import pickle

#training_data, validation_data, test_data = minst_loader.load_data_wrapper() # This will train the network with the normal database


training_data_none, validation_data, test_data = minst_loader.load_data_wrapper() # This will train the network with the extended database
f = gzip.open('expanded_training_data.pkl.gz', 'rb')
training_data = pickle.load(f, encoding="latin1")


model = DigitsRecognizingNN.MLP([784,30,10])
model.SGD(training_data, test_data, epochs=10, mini_batch_size=20, learning_rate = 0.1, lmbd = 25)
