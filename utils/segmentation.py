import numpy as np
import random
from scipy.spatial.distance import squareform, pdist, cdist
from skimage.util import img_as_float
import matplotlib.pyplot as plt
from skimage import transform
from skimage import io
import os

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

def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find np.repeat and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)

    for n in range(num_iters):
        ### YOUR CODE HERE
        #find assignments
        distances = cdist(features, centers, 'euclidean')
        new_assign = np.argmin(distances, axis = 1)
        if np.array_equal(assignments, new_assign):
            break
        assignments = new_assign
        #generate new clusters
        for i in range(len(centers)):
            indexes = [j for j in range(N) if assignments[j] == i]
            centers[i] = np.mean(features[indexes], axis = 0)
        ### END YOUR CODE

    return assignments



def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to defeine distance between two clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """



    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N)
    centers = np.copy(features)
    n_clusters = N

    ###YOUR CODE HERE
    dist = cdist(centers, centers, 'euclidean')
    dist = np.array([[np.inf if i == j else dist[i][j] for i in range(N)] for j in range(N)])
    assigned = {i: [i] for i in range(N)}
    while n_clusters > k:
        #find closest pair of clusters
        ind1, ind2 = np.unravel_index(np.argmin(dist), (N, N))
        #assign all the points in one cluster to another
        #print (n_clusters)
        #print (ind1, ind2, assigned[ind1], assigned[ind2])
        assigned[ind1] = assigned[ind1]+assigned[ind2]
        assigned[ind2] = None
        #print (ind1, ind2, assigned[ind1], assigned[ind2])
        #print ()
        #get a new centroid from the average of the points in the new cluster
        new_centroid = np.mean(features[assigned[ind1]], axis = 0)
        #print (features[assigned[ind1]], new_centroid)
        #make one of the centers found in the minimum distance the centroid and delete the other one
        centers[ind1] = new_centroid
        centers[ind2] = np.full(D, np.inf)
        #update the number of clusters
        n_clusters -= 1
        #find new distances to the new centroid 
        new_dist = cdist([new_centroid], centers, 'euclidean')[0]
        new_dist[ind1] = np.inf
        #print (new_dist)
        #update distance matrix
        #print (dist[0][ind1], dist[0][ind2])
        dist[ind1] = new_dist
        dist[:,ind1] = new_dist
        dist[ind2] = np.full(N, np.inf)
        dist[:,ind2] = np.full(N, np.inf)
        #print (dist[0][ind1], dist[0][ind2])
        #print (dist[ind1])
        
    #convert assigned to assignments
    counter = 0
    for i in assigned:
        if assigned[i] is not None:
            assignments[assigned[i]] = counter
            counter += 1
    ###END YOUR CODE
    return assignments
    

### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    ### YOUR CODE HERE
    H, W = mask_gt.shape
    same = [1 if mask_gt[i][j] == mask[i][j] else 0 for i in range(H) for j in range(W)]
    accuracy = sum(same)/(H*W)
    ### END YOUR CODE

    return accuracy

def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments. 
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy

def visualize_mean_color_image(img, segments):

    img = img_as_float(img)
    k = np.max(segments) + 1
    mean_color_img = np.zeros(img.shape)

    for i in range(k):
        mean_color = np.mean(img[segments == i], axis=0)
        mean_color_img[segments == i] = mean_color

    plt.imshow(mean_color_img)
    plt.axis('off')
    plt.show()

def compute_segmentation(img, k,
        clustering_fn=kmeans_fast,
        feature_fn=color_position_features,
        scale=0):
    """ Compute a segmentation for an image.

    First a feature vector is extracted from each pixel of an image. Next a
    clustering algorithm is applied to the set of all feature vectors. Two
    pixels are assigned to the same segment if and only if their feature
    vectors are assigned to the same cluster.

    Args:
        img - An array of shape (H, W, C) to segment.
        k - The number of segments into which the image should be split.
        clustering_fn - The method to use for clustering. The function should
            take an array of N points and an integer value k as input and
            output an array of N assignments.
        feature_fn - A function used to extract features from the image.
        scale - (OPTIONAL) parameter giving the scale to which the image
            should be in the range 0 < scale <= 1. Setting this argument to a
            smaller value will increase the speed of the clustering algorithm
            but will cause computed segments to be blockier. This setting is
            usually not necessary for kmeans clustering, but when using HAC
            clustering this parameter will probably need to be set to a value
            less than 1.
    """

    assert scale <= 1 and scale >= 0, \
        'Scale should be in the range between 0 and 1'

    H, W, C = img.shape

    if scale > 0:
        # Scale down the image for faster computation.
        img = transform.rescale(img, scale)

    features = feature_fn(img)
    assignments = clustering_fn(features, k)
    segments = assignments.reshape((img.shape[:2]))

    if scale > 0:
        # Resize segmentation back to the image's original size
        segments = transform.resize(segments, (H, W), preserve_range=True)

        # Resizing results in non-interger values of pixels.
        # Round pixel values to the closest interger
        segments = np.rint(segments).astype(int)

    return segments


def load_dataset(data_dir):
    """
    This function assumes 'gt' directory contains ground truth segmentation
    masks for images in 'imgs' dir. The segmentation mask for image
    'imgs/aaa.jpg' is 'gt/aaa.png'
    """

    imgs = []
    gt_masks = []

    # Load all the images under 'data_dir/imgs' and corresponding
    # segmentation masks under 'data_dir/gt'.
    for fname in sorted(os.listdir(os.path.join(data_dir, 'imgs'))):
        if fname.endswith('.jpg'):
            # Load image
            img = io.imread(os.path.join(data_dir, 'imgs', fname))
            imgs.append(img)

            # Load corresponding gt segmentation mask
            mask_fname = fname[:-4] + '.png'
            gt_mask = io.imread(os.path.join(data_dir, 'gt', mask_fname))
            gt_mask = (gt_mask != 0).astype(int) # Convert to binary mask (0s and 1s)
            gt_masks.append(gt_mask)

    return imgs, gt_masks

def segmentation(image, num_segments = 2, scale = 0.1):

    #yung main boy does some segmentation
    img = plt.imread(image)
    segment = compute_segmentation(img, num_segments, clustering_fn=kmeans_fast, feature_fn=color_features, scale=scale)
    
    plt.subplot(1, 2, 1)
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.imshow(segment, cmap='viridis')

    plt.show()

#return the image, but only the parts with in the center segment
def getCenterSegment(image, segmentation, flat = False):
    H,W,C = image.shape
    assert (H,W) == segmentation.shape
    
    center_segment = segmentation[H//2,W//2]
    result = np.zeros(image.shape, dtype=np.uint8)
    result.fill(255)
    idx = (segmentation == center_segment)
    if flat:
        result = image[idx]
    else:
        result[idx] = image[idx]
    return result
