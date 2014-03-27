import pylab as pl
import matplotlib as mpl
import numpy as np

from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM

import csv

class Dataset:

	def __init__ (self, data, target):
		self.data = data
		self.target = target



def make_ellipses(gmm, ax):
    for n, color in enumerate('rgb'):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

data = np.loadtxt('import.csv', delimiter = ',')

for point in data:
	if point[0] == 1:
		point[0] = 0
	if point[0] == 2:
		point[0] = 1
	if point[0] == 4:
		point[0] = 2
	if point[0] == 6:
		point[0] = 3

iris= Dataset(data[:, [1,2]], data[:, [0]])










