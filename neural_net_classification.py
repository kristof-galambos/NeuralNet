"""
own implementation of neural net second try
one hidden layer
arbitrary number of features, arbitrary number of neurons
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score


iris = datasets.load_iris()
X = iris.data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = iris.target
n_classes = np.max(y)+1
y_onehot = np.zeros((len(y), n_classes))
for i in range(len(y)):
    y_onehot[i, y[i]] = 1.
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.3, random_state=101)


# n_classes = 3
# X, y = datasets.make_blobs(n_samples=150, centers=n_classes, n_features=4)
# y_onehot = np.zeros((len(y), n_classes))
# for i in range(len(y)):
#     y_onehot[i, y[i]] = 1.
# X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.3, random_state=101)


                  
class NeuralNetwork():
    
    def __init__(self, learning_rate=1e-1, n_classes=3, n_features=4, n_neurons=10):
        self.learning_rate = learning_rate
        self.weights1 = np.random.randn(n_features, n_neurons)
        self.weights2 = np.random.randn(n_neurons, n_classes)
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
        y_train = np.array(y_train)
        X_train = np.array(X_train)
        for i in range(epochs):
            self.forward_propagate(X_train)
            self.backward_propagate(y_train, X_train)
            
    def predict(self, X_test):
        self.forward_propagate(X_test)
        return self.outputs
            
nn = NeuralNetwork()
nn.train(X_train, y_train, 1000)
y_pred = nn.predict(X_test)
y_test = [np.argmax(x) for x in y_test]
y_pred = [np.argmax(x) for x in y_pred]
print(confusion_matrix(y_test, y_pred))
print('accuracy:', accuracy_score(y_test, y_pred))

# plt.plot(X_test, y_pred, 'r*', label='test data')
# plt.legend()

