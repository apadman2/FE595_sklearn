import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from matplotlib import pyplot
from sklearn.cluster import KMeans
from sklearn import datasets

# SKLearn Assignment - CWID: 10457518
# Question 1
boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target
boston.isnull().sum()

X = boston.drop('MEDV', axis=1)
Y = boston['MEDV']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

# Linear Regression
lin = LinearRegression()
lin.fit(X_train, Y_train)

# Training set
y_pred = lin.predict(X_train)
root_mse = (np.sqrt(mean_squared_error(Y_train, y_pred)))
r_sq = r2_score(Y_train, y_pred)
print('Root Mean Squre Error is {}'.format(root_mse))
print('R-Squared is {}'.format(r_sq))

# Testing set
y_test_pred = lin.predict(X_test)
root_mse_test = (np.sqrt(mean_squared_error(Y_test, y_test_pred)))
r_sq_test = r2_score(Y_test, y_test_pred)
print('Root Mean Square Error {}'.format(root_mse_test))
print('R-Squared is {}'.format(r_sq_test))

importance = lin.coef_
for i, j in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i, j))

# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

# Question 2
iris = datasets.load_iris()
iris_data = pd.DataFrame(iris['data'])
d = []
K = range(1, 10)
for k in K:
    k_means = KMeans(n_clusters=k)
    k_means.fit(iris_data)
    d.append(k_means.inertia_)

plt.figure(figsize=(16, 8))
plt.plot(K, d, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Optimal k')
plt.show()
k_means = KMeans(n_clusters=3)
k_means.fit(iris_data)

wine = datasets.load_wine()
wine_data = pd.DataFrame(wine['data'])
wine_data.head()

d = []
K = range(1, 10)
for k in K:
    k_means = KMeans(n_clusters=k)
    k_means.fit(wine_data)
    d.append(k_means.inertia_)

plt.figure(figsize=(16, 8))
plt.plot(K, d, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Optimal k')
plt.show()

k_means = KMeans(n_clusters=3)
k_means.fit(wine_data)
