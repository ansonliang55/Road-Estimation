"""
Written by: Anson Liang
"""

from skimage import io, color
import slic as sl
from skimage.util import img_as_float
import superpixel as sp
import scipy.io


im_file_name = '../data_road/training/image/um_000000.png'
label_file_name = '../data_road/training/label/um_road_000000.png'
numSegments = 200
com_factor = 10

# read input image
image = img_as_float(io.imread(im_file_name))
label_image = color.rgb2gray(io.imread(label_file_name))#load in grayscale

# get slic superpixel segmentation
#im_sp = sl.getSlicSuperpixels(image, numSegments, com_factor)
im_sp = scipy.io.loadmat('../data_road/training/image/um_000000.mat')['labels'] 

# get superpixel centroid location
sp_location = sp.getSuperPixelLocations(im_sp)

# get superpixel mean color
sp_color = sp.getSuperPixelMeanColor(im_sp, image)

# get superpixel label
sp_label = sp.getSuperPixelLabel(im_sp, label_image, 0.5)
