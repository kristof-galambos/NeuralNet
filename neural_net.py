
import numpy as np


class NeuralNetwork():
    
    def __init__(self, mode, learning_rate=1e-1, n_features=1, n_neurons=8, n_classes=2, activation='sigmoid'):
        self.mode = mode
        self.learning_rate = learning_rate
        self.activation = activation
        self.weights1 = np.random.randn(n_features, n_neurons)
        if self.mode == 'regression':
            self.weights2 = np.random.randn(n_neurons, 1)
        else: #if mode == 'classification'
            self.weights2 = np.random.randn(n_neurons, n_classes)
        self.biases = np.random.randn(n_neurons)
 
    def activation_function(self, z):
        return 1 / (1 + np.exp(-z))
    
    def act_deriv(self, a): #derivative of the activation function
        return a * (1 - a)
    
    def relu(self, z): #rectified linear unit function
        result = np.zeros_like(z)
        for i in range(len(z)):
            for j in range(len(z[0])):
                if z[i, j] > 0:
                    result[i, j] = z[i, j]
        return result
        
    def relu_deriv(self, a): #derivative of the relu activation function
        return a >= 0
    
    def forward_propagate(self, inputs):
        if self.activation == 'sigmoid':
            self.nodes = self.activation_function(np.dot(inputs, self.weights1))
            self.outputs = self.activation_function(np.dot(self.nodes, self.weights2))
        else: # if self.activation == 'relu'
            self.nodes = self.relu(np.dot(inputs, self.weights1))
            self.outputs = self.relu(np.dot(self.nodes, self.weights2))
        
    def backward_propagate(self, y_true, inputs):
        if self.activation == 'sigmoid':
            self.weights2 += self.learning_rate * np.dot(self.nodes.T, (2*(y_true - self.outputs) * self.act_deriv(self.outputs)))
            self.weights1 += self.learning_rate * np.dot(inputs.T,  (np.dot(2*(y_true - self.outputs) * self.act_deriv(self.outputs), self.weights2.T) * self.act_deriv(self.nodes)))
        else: # if self.activation == 'relu'
            self.weights2 += self.learning_rate * np.dot(self.nodes.T, (2*(y_true - self.outputs) * self.relu_deriv(self.outputs)))
            self.weights1 += self.learning_rate * np.dot(inputs.T,  (np.dot(2*(y_true - self.outputs) * self.relu_deriv(self.outputs), self.weights2.T) * self.relu_deriv(self.nodes)))

    def train(self, X_train, y_train, epochs):
        y_train = np.array(y_train)
        if self.mode == 'regression':
            y_train = y_train.reshape((len(y_train),1))
        X_train = np.array(X_train)
        for i in range(epochs):
            self.forward_propagate(X_train)
            self.backward_propagate(y_train, X_train)
            
    def predict(self, X_test):
        self.forward_propagate(X_test)
        if self.mode == 'regression':
            self.outputs = self.outputs.reshape(len(X_test))
        return self.outputs
    
    