import random
import numpy as np
from KNearestNeughbor import KNearestNeighbor
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Load the raw CIFAR-10 data.
print('Loading cifar-10....')
cifar10_dir = 'E:\Python code\my_task1\cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

#  print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

print('Showing the sample pictures of 7*10.....')
plt.ion()
plt.figure(1)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)


print('Reshaping the trainning data into 5000...')
num_training = 5000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

print("trainning data by knn....")
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

dists = classifier.compute_distances_no_loops(X_test)
print(dists.shape)

y_test_pred = classifier.predict_labels(dists, k=1)

# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

print('Predicting  as K has changed from 1 to 5....')
y_test_pred = classifier.predict_labels(dists, k=5)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))


print('choosing diffrent K by cross_validation....')
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []

X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

k_to_accuracies = {}

for k in k_choices:
    k_to_accuracies[k] = np.zeros(num_folds)
    for i in range(num_folds):
        Xtr = np.array(X_train_folds[:i] + X_train_folds[i+1:])
        ytr = np.array(y_train_folds[:i] + y_train_folds[i+1:])
        Xte = np.array(X_train_folds[i])
        yte = np.array(y_train_folds[i])

        Xtr = np.reshape(Xtr, (int(X_train.shape[0] * 4 / 5), -1))
        ytr = np.reshape(ytr, (int(y_train.shape[0] * 4 / 5), -1))
        Xte = np.reshape(Xte, (int(X_train.shape[0] / 5), -1))
        yte = np.reshape(yte, (int(y_train.shape[0] / 5), -1))

        classifier.train(Xtr, ytr)
        yte_pred = classifier.predict(Xte, k)
        yte_pred = np.reshape(yte_pred, (yte_pred.shape[0], -1))
        num_correct = np.sum(yte_pred == yte)
        accuracy = float(num_correct) / len(yte)
        k_to_accuracies[k][i] = accuracy

for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))

print('Visualizing the accuracies....')
plt.figure(2)
for k in k_choices:
  accuracies = k_to_accuracies[k]
  plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')


print('training by best k value')
best_k = 8
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=best_k)

# Compute and display the accuracy
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

plt.ioff()
plt.show()