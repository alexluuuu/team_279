#
# DatasetPrep.py 
# 	
# Contains the methods for file i/o and preliminary dataset processing. 
# 
# 
import pandas as pd
import numpy as np
import subprocess


def PrepareGroundTruth(ground_truth_csv_path, images):
	'''
	the ground truth csv has three columns:
		1) file name of image
		2) whether image shows melanoma
		3) whether image shows seb keratosis

	We want to just take the second column for our binary classifier,
	taking advantage of the sorted image_id column
	'''
	df = pd.read_csv(ground_truth_csv_path)
	groundtruth = [val for (_, image_id, val, _) in df.itertuples() if image_id + '.jpg' in images]
	return groundtruth


def GatherImageSet(imagedir):
	'''
	Use a subprocess to ls into the provided image directory and list the content files. 
	More advanced features can be performed by adding regex to the subprocess, but it is not necessary with our 
	current organization. 

	'''
	   
	sp = subprocess.Popen('ls ' + imagedir, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	image_files = [name.strip() for name in sp.stdout.readlines()]
	return (len(image_files), image_files)


def ParseCommandLine(argvec):
	'''
	Ideal usage: 
	$python mainboy.py ground_truth metadata imagedir 

	Ideally, we would be able to parse the metadata file but we did not. eom. 
	'''
	return {"ground_truth": argvec[1], "imagedir": argvec[3]}


def StoreFeatures(features, out_dest):
    '''
    Given a matrix features, write it to a csv path specified in out_dest. 
    '''
    print "writing features to csv"
    df = pd.DataFrame(features)
    df.to_csv(out_dest, header=None, index=False)


def ReadFeatures(ft_path):
    '''
    Given a path to a file, read it in and return as a matrix.
    '''
    print "reading features from csv"
    df = pd.read_csv(ft_path, header=None)

    return df.as_matrix()
    