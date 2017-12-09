from utils.lbp import *
from utils.DatasetPrep import *
from utils.segmentation import *
from utils.classifier import *
from utils.colors import *
from utils.image_ops import *
from skimage import io
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import sys 


def GatherFeatures(NoI, input_names, imagedir, output_dest_text, output_dest_color, text_ft_path=None, color_ft_path=None):

    texture_features = np.zeros((NoI, 26))
    color_features = np.zeros((NoI, 256))

    if text_ft_path is not None:
        texture_features = ReadFeatures(text_ft_path)
    if color_ft_path is not None:
        color_features = ReadFeatures(color_ft_path)
    else:

        for c, im_file in enumerate(input_names):
            print 'currently processing: ' + im_file
            image = ReadImage(imagedir + im_file)
            texture_features[c,:] = ComputeLBP(image, 3)
            color_features[c, :] = extractColor(image)

        texture_features = normalize(texture_features, 'l1')

        StoreFeatures(texture_features, output_dest_text)
        StoreFeatures(color_features, output_dest_color)

    
    return (texture_features, color_features)


if __name__ == "__main__":

    # get a dictionary of commandline arguments 
    args_dict = ParseCommandLine(sys.argv)

    # find where the groundtruth csv is stored 
    ground_truth_csv_path = args_dict['ground_truth']

    # find where the images are
    imagedir = args_dict['imagedir']

    # Number of Images, list of images
    NoI, input_names = GatherImageSet(imagedir) 

    # Prepare the groundtruth 
    groundtruth = PrepareGroundTruth(ground_truth_csv_path, input_names)
    #print groundtruth

    # find where the images are
    imagedir = args_dict['imagedir']

    # Number of Images, list of images
    NoI, input_names = GatherImageSet(imagedir) 

    texture_features, color_features = GatherFeatures(NoI, input_names, imagedir, "text_ft.csv", "color_ft.csv", text_ft_path="text_ft.csv", color_ft_path="color_ft.csv")

    combined_features = np.concatenate((texture_features, color_features), axis=1)

    #StoreFeatures(combined_features, "combined.txt")
    #StoreFeatures(groundtruth, "groundtruth.txt")

    print "preparing to run classifier"
    print '---------------------------'
    
    
    C = .5
    C_bound = 8
    W_bound = 8
    grid = np.zeros((C_bound, W_bound, 3))
    for i in range(C_bound):
        weight = 1.5
        for j in range(W_bound):
            print 'for C', C
            print 'for weight', weight
            grid[i,j] = EvaluateClassifier(combined_features, groundtruth, C, weight)
            weight += .25
            print '---------------------------'
        C += .5
    print grid

    #RunClassifier(combined_features, groundtruth)

    print 'total number of melanomas in set: ', sum(groundtruth)
    #print "preparing to tune classifier"
   # TuneClassifier(combined_features, groundtruth)
    #TODO: do colors (RGB -> HSV) 

    #TODO: do shape


    #TODO: classifier
