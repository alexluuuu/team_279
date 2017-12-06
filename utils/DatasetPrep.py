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

def PrepareGroundTruth(ground_truth_csv_path, images):

	df = pd.read_csv(ground_truth_csv_path)

	groundtruth = [val for (_, image_id, val, _) in df.itertuples() if image_id + '.jpg' in images]

	return groundtruth



'''

'''
def GatherImageSet(imagedir):
	
	sp = subprocess.Popen('ls ' + imagedir, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	image_files = [name.strip() for name in sp.stdout.readlines()]
	return (len(image_files), image_files)


def ParseCommandLine(argvec):
	'''
	Ideal usage: 

	$python mainboy.py ground_truth metadata imagedir 
	'''
	#if len(argvec) 

	return {"ground_truth": argvec[1], "imagedir": argvec[3]}


def StoreTextureFeatures(texture_features, out_dest):
    print "writing textural features to csv"
    df = pd.DataFrame(texture_features)
    df.to_csv(out_dest)

def ReadTextureFeatures(text_ft_path):
	df = pd.read_csv(text_ft_path, header=None, )

	for vec in df.itertuples():
		print vec
    