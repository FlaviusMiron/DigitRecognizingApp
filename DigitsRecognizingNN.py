"""
A multilayer perceptron, used to learn the digits of the MINST database. Uses mini-batch gradient descent and backpropagation.
It is made for the specific data format given by the "minst_loader.py" data loader and has to be re-adapted for other uses.
"""

import numpy as np
import pickle
import random

class quadratic_cost:
    @staticmethod
    def get_cost_value(output, target):
        return sum((output-target)**2/2)
    
    @staticmethod
    def get_delta(z, a, target):
        return (a - target) * sigmoid_prime(z)
    
class cross_entropy_cost:
    @staticmethod
    def get_cost_value(output, target):
        return np.sum(np.nan_to_num(-target*np.log(output)-(1-target)*np.log(1-output)))
    
    @staticmethod
    def get_delta(z, a, target):
        return a-target

class MLP:
    def __init__(self, sizes, cost = cross_entropy_cost):
        """Initializes various parameters."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost_function = cost
        
        self.biases = [np.random.randn(n,1) for n in self.sizes[1:]]
        self.weights = [np.random.randn(n,m)/np.sqrt(m) for n,m in zip(self.sizes[1:],self.sizes[:-1])]

        self.parameters = []
        self.epoch_performances = []

    def normal_weight_initialization(self):
        """In case you want to try normal initialization(not reccomended)"""
        self.biases = [np.random.randn(n,1) for n in self.sizes[1:]]
        self.weights = [np.random.randn(n,m)for n,m in zip(self.sizes[1:],self.sizes[:-1])]

    def __feed_forward(self, a):
        """Returns the output of the network, given 'a' as input"""
        for w,b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self, training_data, test_data = None, mini_batch_size = 10, epochs = 10 ,learning_rate = 0.5, lmbd = 5):
        """Trains the network using mini-batch gradiend descent."""
        training_data = list(training_data)
        len_training_data = len(training_data)
        print(len_training_data)
        if test_data:
            test_data = list(test_data)
            len_test_data = len(test_data)

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
                self.weights = [(1-learning_rate*(lmbd/len_training_data))*w - (learning_rate/mini_batch_size)*wg for w,wg in zip(self.weights, weights_gradients)]            

            if test_data:
                predicted = self.__evaluate_model(test_data)
                print("Epoch {}: guessed {} out of {}".format(epoch,predicted,len_test_data))
                self.epoch_performances.append(predicted)
                self.parameters.append(([self.weights,self.biases]))


        self.save_data("parameters.pkl")

    def __back_propagate(self, image, target):
        """Returns the gradients of the parameters for a single training example."""
        biases_gradients = [np.zeros((n,1)) for n in self.sizes[1:]]
        weights_gradients = [np.zeros((n,m)) for n,m in zip(self.sizes[1:],self.sizes[:-1])]

        a = image
        activation_list = [a]
        z_values_list=[]

        for w,b in zip(self.weights, self.biases):
            z = np.dot(w,a) + b
            z_values_list.append(z)

            a = sigmoid(z)
            activation_list.append(a)

        delta = self.cost_function.get_delta(z,a,target)
        biases_gradients[-1] = delta
        weights_gradients[-1] = np.dot(delta,activation_list[-2].transpose())

        for l in range(2,len(self.sizes)):
            delta = np.dot(self.weights[-l+1].transpose(),delta) * sigmoid_prime(z_values_list[-l])
            biases_gradients[-l] = delta
            weights_gradients[-l] = np.dot(delta,activation_list[-l-1].transpose())

        return (biases_gradients, weights_gradients)

    def reinitialize_model(self):
        """In case you want to re-train the model with other hyper-parameters."""
        self.biases = [np.random.randn(n,1) for n in self.sizes[1:]]
        self.weights = [np.random.randn(n,m)/np.sqrt(m) for n,m in zip(self.sizes[1:],self.sizes[:-1])]
        self.parameters = []
        self.epoch_performances = []

    def __evaluate_model(self, test_data):
        results = [(np.argmax(self.__feed_forward(image)),target) for image, target in test_data]
        return sum([x == y for (x,y) in results])
    
    def save_data(self, file_name):
        """Used to save the best performing parameters in order for them to be used by the drawing app."""
        file = open(file_name,'wb')
        pickle.dump(self.parameters[np.argmax(self.epoch_performances)],file)
        print(np.argmax(self.epoch_performances))


def sigmoid(z):
        return 1.0 / ( 1.0 + np.exp(-z) )
    
def sigmoid_prime(z):
        return sigmoid(z)*(1 - sigmoid(z))


if __name__ == "__main__":
    print('This model is trained via "model_training.py."')
