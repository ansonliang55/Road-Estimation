"""
Handler for forming data matrix
"""
import superpixel as sp
import slic as sl
from skimage.util import img_as_float
from skimage import io, color
import numpy as np


class Feature:
		def __init__(self):
    		# read input image
				self.image = []
				self.im_sp = []
				self.label_image = []

		def loadImage(self, im_file_name):
				self.image = img_as_float(io.imread(im_file_name))

		def loadSuperpixelImage(self, numSegments, com_factor):
				# get slic superpixel segmentation
				if self.image == []:
						raise Exception("Please load image first")
				self.im_sp = sl.getSlicSuperpixels(self.image, numSegments, com_factor)

		def loadSuperpixelFromFile(self, sp_file_name):
				# get slic superpixel segmentation
				im_sp = scipy.io.loadmat(sp_file_name)['labels'] 

		def loadLabelImage(self, label_file_name):
				self.label_image = color.rgb2gray(io.imread(label_file_name))#load in grayscale

		def getFeaturesVectors(self):

				# get superpixel centroid location
				sp_location = sp.getSuperPixelLocations(self.im_sp)

				# get superpixel mean color		
				sp_color = sp.getSuperPixelMeanColor(self.im_sp, self.image)

				# get superpixel size
				sp_size = sp.getSuperPixelSize(self.im_sp)

				featureVectors = np.vstack((sp_location.T,sp_color.T, sp_size.T)).T
				return featureVectors

		def getLabels(self):

				# get superpixel label
				sp_labels = sp.getSuperPixelLabel(self.im_sp, self.label_image, 0.5)

				return sp_labels

		def getImage(self):
				return self.image

		def getSuperpixelImage(self):
				return self.im_sp

		def getLabelImage(self):
				return self.label_image