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
				self.sp_location = []
				self.sp_color = []
				self.sp_size = []
				self.featureVectors = []
				self.sp_labels = []
				self.edges = []

		def loadImage(self, im_file_name):
				self.image = img_as_float(io.imread(im_file_name))

		def loadSuperpixelImage(self, numSegments, com_factor):
				# get slic superpixel segmentation
				if self.image == []:
						raise Exception("Please load image first")
				self.im_sp = sl.getSlicSuperpixels(self.image, numSegments, com_factor)

		def loadSuperpixelFromFile(self, sp_file_name):
				# get slic superpixel segmentation
				self.im_sp = scipy.io.loadmat(sp_file_name)['labels'] 

		def loadLabelImage(self, label_file_name):
				self.label_image = color.rgb2gray(io.imread(label_file_name))#load in grayscale

		def getEdges(self): 
				#
				self.edges = sp.getPairwiseMatrix(self.im_sp)
				[row, col] = self.edges.shape
				sumDiff = 0
				count = 0
				self.edgesGrad = np.zeros((row, row))
				self.edgesDist = np.zeros((row, row))
				for i in xrange(0,row):
						for j in xrange(i, col):
								if self.edges[i][j]!= 0:
										self.edgesGrad[i][j] = np.linalg.norm(self.sp_color[i] - self.sp_color[j])
										self.edgesGrad[j][i] = self.edgesGrad[i][j]

										self.edgesDist[i][j] = np.linalg.norm(self.sp_location[i] - self.sp_location[j])
										self.edgesDist[j][i] = self.edgesDist[i][j]

										sumDiff += self.edgesGrad[i][j]
										count += 1
				expectColorGrad = sumDiff/count
				#print expectColorGrad
				return self.edges, self.edgesGrad

		def getFeaturesVectors(self):

				# get superpixel centroid location
				self.sp_location = sp.getSuperPixelLocations(self.im_sp)

				# get superpixel mean color		
				self.sp_color = sp.getSuperPixelMeanColor(self.im_sp, self.image)

				# get superpixel size
				self.sp_size = sp.getSuperPixelSize(self.im_sp)

				self.featureVectors = np.vstack((self.sp_location.T,self.sp_color.T, self.sp_size.T)).T
				return self.featureVectors

		def getLabels(self):

				# get superpixel label
				self.sp_labels = sp.getSuperPixelLabel(self.im_sp, self.label_image, 0.5)

				return self.sp_labels

		def getImage(self):
				return self.image

		def getSuperpixelImage(self):
				return self.im_sp

		def getSuperpixelLocation(self):
				return self.sp_location

		def getLabelImage(self):
				return self.label_image