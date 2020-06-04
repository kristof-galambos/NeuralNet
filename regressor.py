

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from neural_net import NeuralNetwork


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

nn = NeuralNetwork(mode='regression')
nn.train(X_train, y_train, 1000)
y_pred = nn.predict(X_test)
print()
r2 = r2_score(y_test, y_pred)
print('The R^2 score is:', r2)

plt.plot(X_test, y_pred, 'r*', label='test data')
plt.legend()

