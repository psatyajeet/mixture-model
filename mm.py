import pylab as pl
import matplotlib as mpl
import numpy as np

from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM

import csv


positions = []
passes = []

data = np.loadtxt('import.csv', delimiter = ',')



# Break up the dataset into non-overlapping training (75%) and testing
# (25%) sets.
skf = StratifiedKFold(data[:, 0], n_folds=4)
# Only take the first fold.
train_index, test_index = next(iter(skf))

X_train = data[:, 1][train_index]
y_train = data[:, 0][train_index]
X_test = data[:, 1][test_index]
y_test = data[:, 0][test_index]


n_classes = len(np.unique(y_train))

# Try GMMs using different types of covariances.
classifiers = dict((covar_type, GMM(n_components=n_classes,
                    covariance_type=covar_type, init_params='wc', n_iter=20))
                   for covar_type in ['spherical', 'diag', 'tied', 'full'])

n_classifiers = len(classifiers)

print n_classes, n_classifiers


for index, (name, classifier) in enumerate(classifiers.iteritems()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    classifier.means_ = np.array([X_train[y_train == i].mean(axis=0)
                                  for i in xrange(n_classes)])

    # Train the other parameters using the EM algorithm.
    classifier.fit(X_train)

    h = pl.subplot(2, n_classifiers / 2, index + 1)
    make_ellipses(classifier, h)

    for n, color in enumerate('rgb'):
        data = iris.data[iris.target == n]
        pl.scatter(data[:, 0], data[:, 1], 0.8, color=color,
                   label=iris.target_names[n])
    # Plot the test data with crosses
    for n, color in enumerate('rgb'):
        data = X_test[y_test == n]
        pl.plot(data[:, 0], data[:, 1], 'x', color=color)

    y_train_pred = classifier.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    pl.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
            transform=h.transAxes)

    y_test_pred = classifier.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    pl.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
            transform=h.transAxes)

    pl.xticks(())
    pl.yticks(())
    pl.title(name)

pl.legend(loc='lower right', prop=dict(size=12))

print train_accuracy
print n_classifiers



