# 
# image_ops.py
# 
# Support the image manipulation operations that we perform throughout the computation process. 
# 

from skimage.transform import rescale
from skimage import io

def ReadImage(img_path):
	'''
	Given the path to an image file, read in the image as a matrix and return a rescaled version of the image. 
	'''
	as_mat = io.imread(img_path)
	return DynamicRescale(as_mat)

def DynamicRescale(img):
	'''
	Given a matrix of shape (h,w,c), rescale based on bucketing of w. 

	We have determined that for extraction of features to take less than 2 hours, large images in the dataset must be 
	aggressively downscaled. 
	'''
	rescale_factor = 1.0
	_, w, _ = img.shape

	if w > 4000:
		rescale_factor = .12 
	elif w > 3000:
		rescale_factor = .14
	elif w > 2000:
		rescale_factor = .2
	elif w > 1000:
		rescale_factor = .3
	elif w > 500:
		rescale_factor = .4

	print 'rescale factor: ', rescale_factor
	return rescale(img, rescale_factor)

def Flatten(img):
	'''
	Flatten an imagee, input as ndarray of shape (h,w,c) to return an array of shape (h,w)
	'''
	h, w, c = img.shape
	return img.reshape((h, w*c))
