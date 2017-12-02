from utils.lbp import *
from utils.DatasetPrep import PrepareGroundTruth

if __name__ == "__main__":

	ground_truth_csv_path = "sample_dataset/ISIC-2017_Training_Part3_GroundTruth.csv"
	
	print "it's me, the main boy"

	scale_adap_boy = SALBP()
	input_names = ['sample_dataset/ISIC_0000000.jpg']
	scale_adap_boy.ComputeLBP(input_names[0])
	print "done"
	#scale_adap_boy.VisualizeLBP()