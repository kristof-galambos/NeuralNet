
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from neural_net import NeuralNetwork


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


# #artificial data
# n_classes = 3
# X, y = datasets.make_blobs(n_samples=150, centers=n_classes, n_features=4)
# y_onehot = np.zeros((len(y), n_classes))
# for i in range(len(y)):
#     y_onehot[i, y[i]] = 1.
# X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.3, random_state=101)

            
nn = NeuralNetwork(mode='classification', n_features=4, n_classes=3)
nn.train(X_train, y_train, 1000)
y_pred = nn.predict(X_test)
y_test = [np.argmax(x) for x in y_test]
y_pred = [np.argmax(x) for x in y_pred]
print(confusion_matrix(y_test, y_pred))
print('accuracy:', accuracy_score(y_test, y_pred))

