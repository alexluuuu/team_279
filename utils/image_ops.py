from skimage.transform import rescale
from skimage import io

def ReadImage(img_path):
	as_mat = io.imread(img_path)
	return DynamicRescale(as_mat)

def DynamicRescale(img):
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
	h, w, c = img.shape
	return img.reshape((h, w*c))
