import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas.io.common
import os

from scipy.io import loadmat

from PIL import Image

from scipy.optimize import minimize

class MultClassClassif():

	def __init__(self, path_data, path_weight, layer_size, num_labels):

		self._df = loadmat(path_data)
		self._w = loadmat(path_weight)
		self._keys = self._df.keys()
		self.X = np.c_[np.ones((self._df['X'].shape[0], 1)), self._df['X']]
		self.y = self._df['y']
		self.theta1 = self._w['Theta1']
		self.theta2 = self._w['Theta2']
		self._l = len(self.y)
		self.layer_size = layer_size
		self.num_labels = num_labels

	def sygmoid(self, data):

		g = 1 / (1 + np.exp(-data))
		return (g)

	def vectCostFunct(self, X, y, theta):

		hypotesis = self.sygmoid(np.dot(X, theta.T))
		log1 = np.log(hypotesis)
		log2 = np.log(1 - hypotesis)
		first_part = np.dot(-y.T, log1)
		second_part = (np.dot((1 - y).T, log2))

		res = first_part - second_part
		summ = np.sum(res)
		return((np.divide(summ, self._l)))

	def vectGradient(self, X, y, theta):

		len_ = len(y)
		hypotesis = self.sygmoid(np.dot(theta.T, X))
		betta = 1 - hypotesis
		res = np.dot(X.T, betta)

		return (np.divide(res, len_))

	def vectRegCostFunc(self, theta, X, y, lambd):

		#ValueError: shapes (401,5000) and (401,5000) not aligned: 5000 (dim 1) != 401 (dim 0) is hell
		len_ = len(y)
		hypotesis = self.sygmoid(np.dot(X, theta))
		log1 = np.log(hypotesis).T
		log2 = np.log(1 - hypotesis).T

		first_part = -1 / len_ * (np.dot(log1, y) + np.dot(log2, 1 - y))
		second_part = (lambd / len_) * np.sum(np.square(theta[1:]))

		res = first_part + second_part

		return (res[0])

	def gradient(self, theta, X, y, lambd):

		len_ = len(y)
		hypotesis = self.sygmoid(np.dot(X, theta.reshape(-1, 1)))

		ans = (1/len_) * np.dot(X.T, hypotesis - y) + (lambd / len_) * np.r_[[[0]], theta[1:].reshape(-1, 1)]

		return (ans.flatten())

	def prediction(self, all_theta, X):

		probs = self.sygmoid(np.dot(X, all_theta.T))

		return (np.argmax(probs, axis=1)+1)

	def oneVsAll(self, X, y, num_labels, lambd):
		
		init_theta = np.zeros((X.shape[1], 1)) #401x1
		all_theta = np.zeros((num_labels, X.shape[1])) # 10x401
		
		print(self.vectRegCostFunc(init_theta, X, y, lambd))
		for c in np.arange(1, num_labels + 1):
			res = minimize(self.vectRegCostFunc, init_theta, args=(X, (y == c)*1, lambd), method=None, jac=self.gradient, options={'maxiter':50})
			all_theta[c - 1] = res.x

		return (all_theta)

	def showRandomImages(self, count=20):

		fig = plt.figure()

		for i in range(1, count+1):
			sample = np.random.choice(self.X.shape[0], count)
			fig.add_subplot(count,1,i)
			plt.imshow(self.X[sample,1:].reshape(-1, 20).T, cmap=None)
			plt.axis('off')
		plt.show()


def main():
	
	#MCC.X - features
	#MCC.y - labels

	layer_size = 400
	num_labels = 10
	path_data, path_weight = os.getcwd() + '/ex3data1.mat', os.getcwd() + '/ex3weights.mat'

	MCC = MultClassClassif(path_data, path_weight, layer_size, num_labels)
	print("theta1 : {}".format(MCC.theta1.shape))
	print("theta2 : {}".format(MCC.theta2.shape))

	MCC.showRandomImages(10)

	lambda_ = 0.1


	# print(MCC.gradient(MCC.X, MCC.y, num_labels, lambda_))
	theta = MCC.oneVsAll(MCC.X, MCC.y, num_labels, lambda_)
	prediction = MCC.prediction(theta, MCC.X)
	print('Accuracy = {0}'.format(np.mean(prediction == MCC.y.ravel()) * 100))
	# print(MCC.vectRegCostFunc(MCC.X, MCC.y, num_labels, lambda_))
	# theta_test = np.array([-2,-1,1,2])
	# X_test = np.concatenate((np.ones((5,1)), np.fromiter((x/10 for x in range(1,16)), float).reshape((3,5)).T), axis = 1)
	# y_test = np.array([1,0,1,0,1])
	# lambda_test = 3

	# # print(MCC.vectRegCostFunc(X_test , y_test, theta_test, lambda_test))
	# print(MCC.gradient(X_test , y_test, theta_test, lambda_test))
	# # print(MCC.gradient(MCC.X, MCC.y, MCC.theta1, 3))

if __name__ == '__main__':
	main()