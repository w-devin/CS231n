import numpy as np

from ImageClassification.NearestNeighborclassifier import NearestNeighbor

if __name__ == '__main__':
	instance = NearestNeighbor()

	X = np.arange(9).reshape(3, 3)
	y = np.array([[1], [2], [3]])

	x = np.array([[1, 2, 3], [4, 5, 6]])
	instance.train(X, y)
	print(instance.predict(x))
