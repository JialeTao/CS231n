import random
import numpy as np


class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    self.X_train = X
    self.y_train = y

  def predict(self, X, k=1, num_loops=0):
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)
    return self.predict_labels(dists, k=k)

  def predict_labels(self, dists, k=1):
      num_test = dists.shape[0]
      y_pred = np.zeros(num_test)
      for i in range(num_test):
          count = 0
          label = 0
          closest_y = []
          idx = np.argsort(dists[i, :], -1)
          closest_y = self.y_train[idx[:k]]
          for j in closest_y:
              tmp = 0
              for kk in closest_y:
                  tmp += (kk == j)
              if tmp > count:
                  count = tmp
                  label = j
          y_pred[i] = label
          # y_pred[i] = np.argmax(np.bincount(closest_y))
      return y_pred

  def compute_distances_no_loops(self, X):
      num_test = X.shape[0]
      num_train = self.X_train.shape[0]
      dists = np.zeros((num_test, num_train))
      dists = np.sqrt(self.getNormMatrix(X, num_train).T + self.getNormMatrix(self.X_train, num_test) - 2 * np.dot(X, self.X_train.T))
      return dists

  def getNormMatrix(self, x, lines_num):
      return np.ones((lines_num, 1)) * np.sum(np.square(x), axis=1)


