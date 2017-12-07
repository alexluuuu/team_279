from utils.lbp import *
from utils.DatasetPrep import *
from utils.segmentation import *
from utils.classifier import *
from utils.colors import *
from skimage import io
import matplotlib.pyplot as plt
import sys 



def GatherTextures(NoI, input_names, imagedir, output_dest, text_ft_path=None):

    texture_features = np.zeros((NoI, 26))


    if text_ft_path is None:
        radius_bounds = (3, 8)
        for c, image in enumerate(input_names):
            print 'currently processing:' + image
            scale_adap_boy = SALBP()
            text_vec = scale_adap_boy.ComputeLBP(3, imagedir + image)
            texture_features[c,:] = text_vec
            #print text_vec
            print "done"
    #        scale_adap_boy.VisualizeLBP()  
        
        StoreFeatures(texture_features, output_dest)
    
    else: 

        texture_features = ReadFeatures(text_ft_path)

        #print texture_features


    return texture_features

def ExtractColors(NoI, input_names, imagedir, output_dest, color_ft_path=None):
    color_features = np.zeros((NoI, 256))

    if color_ft_path is None:
        for c, image in enumerate(input_names):
            print 'currently processing' + imagedir + image
            img_as_mat = io.imread(imagedir+image)
            color_features[c,:] = extractColor(img_as_mat)
            print "done"

        StoreFeatures(color_features, "color_ft.csv")

    else:
        color_features = ReadFeatures(color_ft_path)



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

    # Computation of texture features
    #texture_features = GatherTextures(NoI, input_names, "text_ft.csv", imagedir) #, text_ft_path="text_ft.csv")

    # Computation of color features
    color_features = ExtractColors(NoI, input_names, imagedir, "color_ft.csv")

    print color_features
    #combined_features = np.concatenate((texture_features, color_features), axis=1)


    #RunClassifier(texture_features, groundtruth )
    TuneClassifier(texture_features, groundtruth)
    #TODO: do colors (RGB -> HSV) 

    #TODO: do shape


    #TODO: classifier
