'''
EXTRACTION OF COLOR-RELATED FEATURES
'''

import numpy as np
from skimage.util import img_as_float

### Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))

    ### YOUR CODE HERE
    features = img.reshape((H*W, C))
    ### END YOUR CODE

    return features

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).
    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    ### YOUR CODE HERE
    features = np.array([[img[i][j][0], img[i][j][1], img[i][j][2], float(i), float(j)] for i in range(H) for j in range(W)])
    means = np.mean(features, axis = 0)
    stdevs = np.std(features, axis = 0)
    features = np.subtract(features, means)
    features = np.divide(features, stdevs)
    
    ### END YOUR CODE

    return features