from utils.lbp import *
from utils.DatasetPrep import *
from utils.segmentation import *
import matplotlib.pyplot as plt

if __name__ == "__main__":

    ground_truth_csv_path = "sample_dataset/ISIC-2017_Training_Part3_GroundTruth.csv"
    
    print "it's me, the main boy"

    scale_adap_boy = SALBP()
    input_names = ['sample_dataset/images/ISIC_0000000.jpg']
    scale_adap_boy.ComputeLBP(input_names[0])
    print "done"
    #scale_adap_boy.VisualizeLBP()
    
    #yung main boy does some segmentation
    img = plt.imread('sample_dataset/images/ISIC_0000000.jpg')
    segment = compute_segmentation(img, 2, clustering_fn=kmeans_fast, feature_fn=color_features, scale=0.1)
    
    plt.subplot(1, 2, 1)
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.imshow(segment, cmap='viridis')

    plt.show()