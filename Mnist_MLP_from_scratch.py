import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import fetch_openml

from sklearn.model_selection import train_test_split
X, Y = fetch_openml('mnist_784', version=1, return_X_y=True)
Y = Y.astype(int)
X = ((X / 255.) - .5) * 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=123, stratify=y)
training_d = [(x, y) for x, y in zip(X_train, y_train)]
testing_d = [(x, y) for x, y in zip(X_test, y_test)]


# this is the class for the multy layer perceptron
class MultyLayerPerceptron:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.layers_no = len(layer_sizes)
        self.biases = [np.random.randn(n, 1) for n in self.layer_sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

    # this is the sigmoid function
    def sigmoid(self, z):
        a = (1.0/(1.0 + np.exp(-z)))
        return a

    # this is the derivative of the sigmoid function
    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1 - self.sigmoid(z))

    # the derivative of the loss function
    def cost_derivative(self, out_y, y):
        return out_y-y

    # this is the feedforward propagation function of the NN
    def feedforward(self, a):
        """forward propagate through all the layers"""
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid((np.dot(w, a) + b))

        return a

    # the back propagation method
    def backpropagation(self, inputs, l2):
        x, y = inputs
        w_grads = [np.zeros(w.shape) for w in self.weights]
        b_grads = [np.zeros(b.shape) for b in self.biases]
        activations = [x]
        a = x
        weights = self.weights
        z_s = []

        for w, b in zip(self.weights, self.biases):
            z = (np.dot(w, a) + b)
            z_s.append(z)
            a = self.sigmoid(z)
            activations.append(a)

        output_error = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(z_s[-1])
        b_grads[-1] = output_error
        w_grads[-1] = np.dot(output_error, activations[-2])

        for l in range(2, self.layers_no):
            output_error = np.dot(weights[-l + 1].transpose(), output_error) * self.sigmoid_prime(z_s[-l])
            b_grads[-l] = output_error
            w_grads[-l] = np.dot(output_error, activations[-l-1])

        # Add L2 regularization to the gradients
        w_grads = [wg + (l2 / len(inputs[0])) * w for wg, w in zip(w_grads, self.weights)]

        return w_grads, b_grads

    # this is for updating the parameters
    def update_parameters(self, training_batch, eta, l2):
        total_w_grads = [np.zeros(w.shape) for w in self.weights]
        total_b_grads = [np.zeros(b.shape) for b in self.biases]

        for x in training_batch:
            w_grad, b_grad = self.backpropagation(x, l2)
            total_b_grads = [tbg + bg for tbg, bg in zip(total_b_grads, b_grad)]
            total_w_grads = [twg + wg for twg, wg in zip(total_w_grads, w_grad)]
            self.weights = [w - (eta/len(training_batch))* nw for w, nw in zip(self.weights, total_w_grads)]
            self.biases = [b - (eta/len(training_batch))* nb for b, nb in zip(self.biases, total_b_grads)]

    # the fitting method
    def fit(self, training_data, epochs, eta, minibatch_size, l2,  test_data=None):
        training_size = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+minibatch_size] for k in range(0, training_size, minibatch_size)]
            for mini_batch in mini_batches:
                self.update_parameters(mini_batch, eta, l2)

            # testing during training
            print(f"completed epoch {epoch}")
            if test_data:
                self.evaluate(test_data)
            else:
                print("----------------------------")
            print("###################################")

    # predicting method
    def predict(self, inputs):
        final_a = self.feedforward(inputs)
        predicted_value = np.argmax(final_a)
        return predicted_value

    # evaluation
    def evaluate(self, test_data):
        correct = 0
        for x, y in test_data:
            if self.predict(x) == y:
                correct += 1

        print(f"the percentage success rate is {correct/len(test_data)}%")

    """
    testing during the fitting --> 2 done
    evaluating -->3 done
    visualizing the learning --> 5
    add a regularization term --> 1 done
    add a prediction method --> 4 done
    """


net = MultyLayerPerceptron([784, 30, 10])
net.fit(training_d, 30, 0.001, 100, 0.01, testing_d)