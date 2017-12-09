# 
# lbp.py
# 
# Implementation of wrappers for linear binary pattern computation using 
# the scikit-image package. Includes tools for visualizing LBP and transforming 
# a LBP matrix into a vector of counts. 
# 
# 
# 

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
from sklearn.preprocessing import normalize
from image_ops import *

from scipy import ndimage


def ComputeLBP(image, radius):
    n_points = 8*radius
    lbp = local_binary_pattern( Flatten(image), n_points, radius, 'uniform')
    return ConvertToCounts(lbp.ravel())

def ConvertToCounts(vec):
    count_vec = np.zeros(26)
    for local_lbp in vec:
        count_vec[int(local_lbp)] += 1
    return count_vec

def VisualizeLBP(radius=3, ):

    '''
    Visualize the local binary patterns as overlays on the image 
    '''
    n_points = 8*radius
    fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
    plt.gray()

    titles = ('edge', 'flat', 'corner')
    w = width = radius - 1
    edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
    flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
    i_14 = n_points // 4            # 1/4th of the histogram
    i_34 = 3 * (n_points // 4)      # 3/4th of the histogram
    corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
                     list(range(i_34 - w, i_34 + w + 1)))

    label_sets = (edge_labels, flat_labels, corner_labels)

    for ax, labels in zip(ax_img, label_sets):
        #print labels.shape
        #print self.img.shape
        ax.imshow(self._overlay_labels(labels))

    for ax, labels, name in zip(ax_hist, label_sets, titles):
        counts, _, bars = hist(ax)
        highlight_bars(bars, labels)
        ax.set_ylim(ymax=np.max(counts[:-1]))
        ax.set_xlim(xmax=n_points + 2)
        ax.set_title(name)

    ax_hist[0].set_ylabel('Percentage')
    for ax in ax_img:
        ax.axis('off')

    plt.show()


 def overlay_labels(img, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=img.astype(int), bg_label=0, alpha=0.5)

def highlight_bars(bars, indexes):
    '''
    Change some colors in the bar graph histogram 
    '''
    for i in indexes:
        bars[i].set_facecolor('r')


def ConvertToCounts(self, text_vec):
    count_vec = np.zeros(26)
    for local_lbp in text_vec:
        count_vec[int(local_lbp)] += 1
    return count_vec

def hist(self, ax):
    n_bins = int(self.lbp.max() + 1)
    return ax.hist(self.lbp.ravel(), normed = True, bins=n_bins, range=(0, n_bins), 
        facecolor='0.5')






