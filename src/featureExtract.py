"""
Handler for forming data matrix
"""
import superpixel as sp
import slic as sl
from skimage.util import img_as_float
from skimage import io, color
import numpy as np

def getFeaturesVectors(im_file_name, sp_file_names):
		numSegments = 200
		com_factor = 10
		# read input image
		image = img_as_float(io.imread(im_file_name))

		# get slic superpixel segmentation
		im_sp = sl.getSlicSuperpixels(image, numSegments, com_factor)
		#im_sp = scipy.io.loadmat(sp_file_name)['labels'] 

		# get superpixel centroid location
		sp_location = sp.getSuperPixelLocations(im_sp)

		# get superpixel mean color		
		sp_color = sp.getSuperPixelMeanColor(im_sp, image)

		# get superpixel size
		sp_size = sp.getSuperPixelSize(im_sp)

		featureVectors = np.vstack((sp_location.T,sp_color.T, sp_size.T)).T
		return featureVectors, im_sp, image

def getLabels(label_file_name, im_sp):

		label_image = color.rgb2gray(io.imread(label_file_name))#load in grayscale
		# get superpixel label
		sp_labels = sp.getSuperPixelLabel(im_sp, label_image, 0.5)

		return sp_labels