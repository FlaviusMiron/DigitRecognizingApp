"""
A multilayer perceptron, used to learn the digits of the MINST database. Uses mini-batch gradient descent and backpropagation.
It is made for the specific data format given by the "minst_loader.py" data loader and has to be re-adapted for other uses.
"""

import numpy as np
import pickle
import random

class MLP:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        
        self.biases = [np.random.randn(n,1) for n in self.sizes[1:]]
        self.weights = [np.random.randn(n,m) for n,m in zip(self.sizes[1:],self.sizes[:-1])]

        self.parameters = []
        self.epoch_performances = []

    def __feed_forward(self, a):
        for w,b in zip(self.weights, self.biases):
            a = self.__sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self, training_data, test_data = None, mini_batch_size = 10, epochs = 10 ,learning_rate = 0.5):
        training_data = list(training_data)
        len_training_data = len(training_data)
        if test_data:
            test_data = list(test_data)

        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,len_training_data,mini_batch_size)]

            for mini_batch in mini_batches:

                biases_gradients = [np.zeros((n,1)) for n in self.sizes[1:]]
                weights_gradients = [np.zeros((n,m)) for n,m in zip(self.sizes[1:],self.sizes[:-1])]


                for image, target in mini_batch:
                    single_biases_gradients, single_weights_gradients = self.__back_propagate(image, target)

                    biases_gradients = [bg + sbg for bg, sbg in zip(biases_gradients,single_biases_gradients)]
                    weights_gradients = [wg + swg for wg, swg in zip(weights_gradients,single_weights_gradients)]

                self.biases = [b - (learning_rate/mini_batch_size)*bg for b,bg in zip(self.biases, biases_gradients)]
                self.weights = [w - (learning_rate/mini_batch_size)*wg for w,wg in zip(self.weights, weights_gradients)]            

            if test_data:
                predicted = self.__evaluate_model(test_data)
                print("Epoch {}: guessed {} out of {}".format(epoch,predicted,len(test_data)))
                self.epoch_performances.append(predicted)
                self.parameters.append(([self.weights,self.biases]))


        self.save_data("parameters.pkl")

    def __back_propagate(self, image, target):
        biases_gradients = [np.zeros((n,1)) for n in self.sizes[1:]]
        weights_gradients = [np.zeros((n,m)) for n,m in zip(self.sizes[1:],self.sizes[:-1])]

        a = image
        activation_list = [a]
        z_values_list=[]

        for w,b in zip(self.weights, self.biases):
            z = np.dot(w,a) + b
            z_values_list.append(z)

            a = self.__sigmoid(z)
            activation_list.append(a)

        delta = self.__cost_derivative(a, target) * self.__sigmoid_prime(z)
        biases_gradients[-1] = delta
        weights_gradients[-1] = np.dot(delta,activation_list[-2].transpose())

        for l in range(2,len(self.sizes)):
            delta = np.dot(self.weights[-l+1].transpose(),delta) * self.__sigmoid_prime(z_values_list[-l])
            biases_gradients[-l] = delta
            weights_gradients[-l] = np.dot(delta,activation_list[-l-1].transpose())

        return (biases_gradients, weights_gradients)

    def reinitialize_model(self):
        self.biases = [np.random.randn(n,1) for n in self.sizes[1:]]
        self.weights = [np.random.randn(n,m) for n,m in zip(self.sizes[1:],self.sizes[:-1])]

    def __evaluate_model(self, test_data):
        results = [(np.argmax(self.__feed_forward(image)),target) for image, target in test_data]
        return sum([x == y for (x,y) in results])
    
    def save_data(self, file_name):
        file = open(file_name,'wb')
        pickle.dump(self.parameters[np.argmax(self.epoch_performances)],file)
        print(np.argmax(self.epoch_performances))

    def __cost_derivative(self, activation, target):
        return activation - target

    def __sigmoid(self, z):
        return 1.0 / ( 1.0 + np.exp(-z) )
    
    def __sigmoid_prime(self, z):
        return self.__sigmoid(z)*(1 - self.__sigmoid(z))


if __name__ == "__main__":
    print('This model is trained via "model_training.py."')
