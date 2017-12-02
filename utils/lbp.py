import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb

from scipy import ndimage


class SALBP():

    def __init__(self):
        self.radius = 3
        self.n_points = 8*self.radius
        self.METHOD = 'uniform'
        self.linbp = None
        self.img = None

    def _overlay_labels(self, labels):
        mask = np.logical_or.reduce([self.linbp == each for each in labels])
        print self.img.shape
        print self.linbp.shape
        print self.img
#        print labels
        return label2rgb(mask, image=self.img, bg_label=0, alpha=0.5)

    def _highlight_bars(self, bars, indexes):
        for i in indexes:
            bars[i].set_facecolor('r')

    def ComputeLBP(self, image_str):
        self.img = plt.imread(image_str).astype(int)
        if len(self.img.shape) > 2:
            print "Flattening iamge..."
            self.img = ndimage.imread(image_str, flatten=True, mode='RGB').astype(int)

        self.linbp = local_binary_pattern(self.img, self.n_points, self.radius, self.METHOD)

    def _hist(self, ax):
        n_bins = int(self.linbp.max() + 1)
        return ax.hist(self.linbp.ravel(), normed = True, bins=n_bins, range=(0, n_bins), 
            facecolor='0.5')


    def VisualizeLBP(self):
        fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
        plt.gray()

        titles = ('edge', 'flat', 'corner')
        w = width = self.radius - 1
        edge_labels = range(self.n_points // 2 - w, self.n_points // 2 + w + 1)
        flat_labels = list(range(0, w + 1)) + list(range(self.n_points - w, self.n_points + 2))
        i_14 = self.n_points // 4            # 1/4th of the histogram
        i_34 = 3 * (self.n_points // 4)      # 3/4th of the histogram
        corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
                         list(range(i_34 - w, i_34 + w + 1)))

        label_sets = (edge_labels, flat_labels, corner_labels)

        for ax, labels in zip(ax_img, label_sets):
            #print labels.shape
            #print self.img.shape
            ax.imshow(self._overlay_labels(labels))

        for ax, labels, name in zip(ax_hist, label_sets, titles):
            counts, _, bars = self._hist(ax)
            self._highlight_bars(bars, labels)
            ax.set_ylim(ymax=np.max(counts[:-1]))
            ax.set_xlim(xmax=self.n_points + 2)
            ax.set_title(name)

        ax_hist[0].set_ylabel('Percentage')
        for ax in ax_img:
            ax.axis('off')

        plt.show()

'''
# settings for LBP
radius = 3
n_points = 8 * radius
METHOD = 'uniform'

def overlay_labels(image, linbp, labels):
    mask = np.logical_or.reduce([linbp == each for each in labels])
    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)


def highlight_bars(bars, indexes):
    for i in indexes:
        bars[i].set_facecolor('r')


image = data.load('brick.png')
linbp = local_binary_pattern(image, n_points, radius, METHOD)


def hist(ax, linbp):
    n_bins = int(linbp.max() + 1)
    return ax.hist(linbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')


# plot histograms of LBP of textures
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
    ax.imshow(overlay_labels(image, linbp, labels))

for ax, labels, name in zip(ax_hist, label_sets, titles):
    counts, _, bars = hist(ax, linbp)
    highlight_bars(bars, labels)
    ax.set_ylim(ymax=np.max(counts[:-1]))
    ax.set_xlim(xmax=n_points + 2)
    ax.set_title(name)

ax_hist[0].set_ylabel('Percentage')
for ax in ax_img:
    ax.axis('off')

plt.show()
'''


