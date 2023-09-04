"""
Script that trains the multi-layer perceptron.
Will print the accuracy on the unseen, test data.
"""

import minst_loader
import DigitsRecognizingNN

training_data, validation_data, test_data = minst_loader.load_data_wrapper()

model = DigitsRecognizingNN.MLP([784,30,10])
model.SGD(training_data, test_data, epochs=10, mini_batch_size=10, learning_rate = 2.5)
