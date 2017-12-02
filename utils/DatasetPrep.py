import pandas as pd
import numpy as np

'''
the ground truth csv has three columns:
1) file name of image
2) whether image shows melanoma
3) whether image shows seb keratosis

We want to just take the second column for our binary classifier,
taking advantage of the sorted image_id column
'''

def PrepareGroundTruth(ground_truth_csv_path):
	ground_truth_csv_path = "sample_dataset/ISIC-2017_Training_Part3_GroundTruth.csv"

	df = pd.read_csv(ground_truth_csv_path)
	data_as_mat = df.as_matrix()
	return data_as_mat[:, 1]

