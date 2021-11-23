from sklearn import datasets
from sklearn.utils import shuffle

def load_IRIS(test=True):
	iris = datasets.load_iris()
	X, y = shuffle(iris.data, iris.target, random_state= 1230)
	if test:
		X_train = X[:100, :]
		y_train = y[:100]
		X_test = X[100:, :]
		y_test = y[100:]
		return X_train, y_train, X_test, y_test
	else:
		X = iris.data[:, :]
		y = iris.target
		return X, y