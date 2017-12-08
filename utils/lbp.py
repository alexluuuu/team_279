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

class SALBP():
    '''
    Class which encapsulates computation of local binary patterns and visualization of LBP. 
    '''

    def __init__(self):
        
        self.METHOD = 'uniform'
        self.lbp = None

    
    def ComputeLBP(self, radius, image_str):
        
        n_points = 8*radius
        img = self._ReadFlattenImage(image_str)
        img = DynamicRescale(img)
        self.lbp = local_binary_pattern(img, n_points, radius, self.METHOD)

        return self._ConvertToCounts(self.lbp.ravel())

    def VisualizeLBP(self, radius):

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
            counts, _, bars = self._hist(ax)
            self._highlight_bars(bars, labels)
            ax.set_ylim(ymax=np.max(counts[:-1]))
            ax.set_xlim(xmax=n_points + 2)
            ax.set_title(name)

        ax_hist[0].set_ylabel('Percentage')
        for ax in ax_img:
            ax.axis('off')

        plt.show()


     # def _overlay_labels(self, labels):
     #    mask = np.logical_or.reduce([self.lbp == each for each in labels])
     #    return label2rgb(mask, image=self.img.astype(int), bg_label=0, alpha=0.5)

    def _highlight_bars(self, bars, indexes):
        '''
        Change some colors in the bar graph histogram 
        '''
        for i in indexes:
            bars[i].set_facecolor('r')

    def _ReadFlattenImage(self, image_str):
        '''
        Read in the image and flatten to 2 dimensions if necessary
        '''
        img = plt.imread(image_str)
        if len(img.shape) > 2:
            #print "Flattening image..."
            img = ndimage.imread(image_str, flatten=True, mode='RGB').astype(int)

        return img


    def _ConvertToCounts(self, text_vec):
        count_vec = np.zeros(26)
        for local_lbp in text_vec:
            count_vec[int(local_lbp)] += 1
        return count_vec

    def _hist(self, ax):
        n_bins = int(self.lbp.max() + 1)
        return ax.hist(self.lbp.ravel(), normed = True, bins=n_bins, range=(0, n_bins), 
            facecolor='0.5')






