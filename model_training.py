"""
Script that trains the multi-layer perceptron.
Will print the accuracy on the unseen, test data.
Can be trained either with the normal or the extended database (Note: the extended database takes significantly more time to train)
Comment the corresponding lines to chose the training data. Right now it uses the normal database.
"""

import minst_loader
import extended_minst_loader
import DigitsRecognizingNN

training_data, validation_data, test_data = minst_loader.load_data_wrapper() # This will train the network with the normal database

# training_data, validation_data, test_data = extended_minst_loader.load_data_wrapper() # This will train the network with the extended database


model = DigitsRecognizingNN.MLP([784,10,10])
model.SGD(training_data, test_data, epochs=10, mini_batch_size=10, learning_rate = 0.1, lmbd = 5)
