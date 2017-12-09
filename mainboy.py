# 
# mainboy.py
# 
# Our main method -- contains the full workflow for gathering images, processing them, and tuning a classifier. 
# Intermediate steps can be saved to file for efficiency and safety. 
# 
# 

from utils.DatasetPrep import *
from utils.classifier import *
from utils.feature_extractor import *
import sys 


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

    # Gather/compute the features
    texture_features, color_features = GatherFeatures(NoI, input_names, imagedir, "text_ft.csv", "color_ft.csv", text_ft_path="text_ft.csv", color_ft_path="color_ft.csv")

    # Combine the features
    combined_features = np.concatenate((texture_features, color_features), axis=1)  

    # Tune classifier
    Tuner(combined_features, groundtruth)
