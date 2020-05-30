"""
own implementation of neural net second try
one hidden layer
arbitrary number of features, arbitrary number of neurons
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler


#1D artificial data
X = np.arange(0, 20).reshape(20, 1) + np.random.randn(20, 1)
y = (np.arange(0, 20) + np.random.randn(20)).reshape(20, 1)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
plt.plot(X, y, '*', label='train data')
plt.xlabel('features')
plt.ylabel('labels')
plt.title('DNN Regressor')


class NeuralNetwork():
    
    def __init__(self, learning_rate=1e-1, n_features=1, n_neurons=8):
        self.learning_rate = learning_rate
        self.weights1 = np.random.randn(n_features, n_neurons)
        self.weights2 = np.random.randn(n_neurons, 1)
        self.biases = np.random.randn(n_neurons)
 
    def activation_function(self, z):
        return 1 / (1 + np.exp(-z))
    
    def act_derivative(self, a): #derivative of the activation function
        return a * (1 - a)
    
    def forward_propagate(self, inputs):
        self.nodes = self.activation_function(np.dot(inputs, self.weights1))
        self.outputs = self.activation_function(np.dot(self.nodes, self.weights2))
        
    def backward_propagate(self, y_true, inputs):
        self.weights2 += self.learning_rate * np.dot(self.nodes.T, (2*(y_true - self.outputs) * self.act_derivative(self.outputs)))
        self.weights1 += self.learning_rate * np.dot(inputs.T,  (np.dot(2*(y_true - self.outputs) * self.act_derivative(self.outputs), self.weights2.T) * self.act_derivative(self.nodes)))

    def train(self, X_train, y_train, epochs):
        y_train = np.array(y_train).reshape((len(y_train),1))
        X_train = np.array(X_train)
        for i in range(epochs):
            self.forward_propagate(X_train)
            self.backward_propagate(y_train, X_train)
            
    def predict(self, X_test):
        self.forward_propagate(X_test)
        return self.outputs.reshape(len(X_test))
            
nn = NeuralNetwork()
nn.train(X_train, y_train, 1000)
y_pred = nn.predict(X_test)
print()
r2 = r2_score(y_test, y_pred)
print('The R^2 score is:', r2)

plt.plot(X_test, y_pred, 'r*', label='test data')
plt.legend()

