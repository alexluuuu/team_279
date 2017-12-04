import pandas as pd
import numpy as np
import subprocess

'''
the ground truth csv has three columns:
1) file name of image
2) whether image shows melanoma
3) whether image shows seb keratosis

We want to just take the second column for our binary classifier,
taking advantage of the sorted image_id column
'''

def PrepareGroundTruth(ground_truth_csv_path):

	df = pd.read_csv(ground_truth_csv_path)
	data_as_mat = df.as_matrix()
	return data_as_mat[:, 1]


'''

'''
def GatherImageSet(ImageDir):
	
	sp = subprocess.Popen('ls ' + ImageDir, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	image_files = dummy.stdout.readlines()
	return (len(image_files), image_files)
