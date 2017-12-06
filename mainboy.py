from utils.lbp import *
from utils.DatasetPrep import *
from utils.segmentation import *
from utils.classifier import *
import matplotlib.pyplot as plt
import sys 


def segmentation(image):

    #yung main boy does some segmentation
    img = plt.imread(image)
    segment = compute_segmentation(img, 2, clustering_fn=kmeans_fast, feature_fn=color_features, scale=0.1)
    
    plt.subplot(1, 2, 1)
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.imshow(segment, cmap='viridis')

    plt.show()




if __name__ == "__main__":

    #get a dictionary of commandline arguments 
    args_dict = ParseCommandLine(sys.argv)

    #find where the groundtruth csv is stored 
    ground_truth_csv_path = args_dict['ground_truth']



    #find where the images are
    imagedir = args_dict['imagedir']

    #Number of Images, list of images
    NoI, input_names = GatherImageSet(imagedir) 


    #Prepare the groundtruth 
    groundtruth = PrepareGroundTruth(ground_truth_csv_path, input_names)
    #print groundtruth

    #find where the images are
    imagedir = args_dict['imagedir']

    #Number of Images, list of images
    NoI, input_names = GatherImageSet(imagedir) 

    #Computation of texture features

    texture_features = np.zeros((5, 26))
    radius_bounds = (3, 8)

    for c, image in enumerate(input_names[:5]):
        print 'currently processing:' + image
        scale_adap_boy = SALBP()
        #text_vec = scale_adap_boy.ComputeScaleAdaptive(radius_bounds, imagedir + image)
        text_vec = scale_adap_boy.ComputeLBP(3, imagedir + image)
        texture_features[c,:] = text_vec
        #print text_vec
        print "done"
#        scale_adap_boy.VisualizeLBP()
        #segmentation(image)

    print texture_features
    StoreTextureFeatures(texture_features, "text_ft.csv")

    ReadTextureFeatures("text_ft.csv")
    #RunClassifier(texture_features, groundtruth[:50])
    #TODO: do colors (RGB -> HSV) 

    #TODO: do shape


    #TODO: classifier


    #yung main boy does some segmentation
    # img = plt.imread('sample_dataset/images/ISIC_0000000.jpg')
    # segment = compute_segmentation(img, 2, clustering_fn=kmeans_fast, feature_fn=color_features, scale=0.1)
    
    # plt.subplot(1, 2, 1)
    # plt.imshow(img)

    # plt.subplot(1, 2, 2)
    # plt.imshow(segment, cmap='viridis')

    # plt.show()