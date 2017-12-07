'''
EXTRACTION OF COLOR-RELATED FEATURES
'''

import numpy as np
from segmentation import *
from skimage.color import rgb2hsv

HUE = 180
SATURATION = 256
VALUE = 256

#extract the counts
def extractColor(image, colorspace = 'RGB'):
    image = rgb2hsv(image)
    segment = compute_segmentation(image, 3, clustering_fn=kmeans_fast, \
                                   feature_fn=color_position_features, scale=0.1)
    center = getCenterSegment(image, segment, flat = True)
    counts = np.zeros((16, 4, 4))
    for i in center:
        idx = [int(i[0]*16), int(i[1]*4), int(i[2]*4)]
        if idx[0] == 16:
            idx[0] = 15
        if idx[1] == 4:
            idx[1] = 3
        if idx[2] == 4:
            idx[2] = 3
        counts[tuple(idx)] += 1
    counts = counts/len(center)
    return np.ravel(counts)