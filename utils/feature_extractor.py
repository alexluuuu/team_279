from lbp import *
from DatasetPrep import *
from utils.colors import *
from utils.image_ops import *





def GatherFeatures(NoI, input_names, imagedir, output_dest_text, output_dest_color, text_ft_path=None, color_ft_path=None):

    texture_features = np.zeros((NoI, 26))
    color_features = np.zeros((NoI, 256))

    if text_ft_path is not None:
        texture_features = ReadFeatures(text_ft_path)
    if color_ft_path is not None:
        color_features = ReadFeatures(color_ft_path)
    else:

        for c, im_file in enumerate(input_names):
            print 'currently processing: ' + im_file
            image = ReadImage(imagedir + im_file)
            texture_features[c,:] = ComputeLBP(image, 3)
            color_features[c, :] = extractColor(image)

        texture_features = normalize(texture_features, 'l1')

        StoreFeatures(texture_features, output_dest_text)
        StoreFeatures(color_features, output_dest_color)

    
    return (texture_features, color_features)
