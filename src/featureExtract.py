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
        self.im = []
        self.im_gray = []
        self.im_sp = []
        self.label_im = []
        self.sp_location = []
        self.sp_color = []
        self.sp_size = []
        self.sp_shape = []
        self.sp_hog = []
        self.sp_color_hist = []
        self.featureVectors = []
        self.sp_labels = []
        self.edges = []
        self.edge_featureVectors = []

    def loadImage(self, im_file_name):
        self.im = img_as_float(io.imread(im_file_name))
        self.im_gray = color.rgb2gray(self.im)

    def loadSuperpixelImage(self):
        # get slic superpixel segmentation
        if self.im == []:
            raise Exception("Please load image first")
        #300, 3
        self.im_sp = sl.getSlicSuperpixels(self.im, 400, 3)

    def loadSuperpixelFromFile(self, sp_file_name):
        # get slic superpixel segmentation
        self.im_sp = scipy.io.loadmat(sp_file_name)['labels'] 

    def loadLabelImage(self, label_file_name):
        self.label_im = color.rgb2gray(io.imread(label_file_name))#load in grayscale

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
                    #self.edge_featureVectors.append([self.edgesGrad[i][j], self.edgesDist[i][j]])
                    sumDiff += self.edgesGrad[i][j]
                    count += 1
        expectColorGrad = sumDiff/count
        #print self.edge_featureVectors[:][0]
        #print expectColorGrad
        return self.edges, self.edgesGrad, self.edgesDist


    def getFeaturesVectors(self):

        # get superpixel centroid location
        self.sp_location = sp.getSuperPixelLocations(self.im_sp)

        # get superpixel mean color    
        self.sp_color = sp.getSuperPixelMeanColor(self.im_sp, self.im)

        # get superpixel shape
        #self.sp_shape = sp.getSuperPixelShape(self.im_sp)

        # get superpixel histogram of oriented gradient
        self.sp_hog = sp.getSuperPixelOrientedHistogram(self.im_sp, self.im_gray)

        #get superpixel histogram of color
        #self.sp_color_hist = sp.getSuperPixelColorHistogram(self.im_sp, self.im)

        # get superpixel size
        self.sp_size = sp.getSuperPixelSize(self.im_sp)

        # all
        self.featureVectors = np.vstack((self.sp_location.T, self.sp_color.T, self.sp_size.T, self.sp_hog.T)).T
        # basic feature with hog
        #self.featureVectors = np.vstack((self.sp_location.T,self.sp_color.T, self.sp_size.T, self.sp_hog.T)).T

        # basic feature vector
        #self.featureVectors = np.vstack((self.sp_location.T,self.sp_color.T, self.sp_size.T)).T

        # basic feature vector no size
        #self.featureVectors = np.vstack((self.sp_location.T,self.sp_color.T)).T

        return self.featureVectors

    def getSuperPixelLabels(self):

        # get superpixel label
        self.sp_labels = sp.getSuperPixelLabel(self.im_sp, self.label_im, 0.5)

        return self.sp_labels

    def getImage(self):
        return self.im

    def getSuperpixelImage(self):
        return self.im_sp

    def getSuperpixelLocation(self):
        return self.sp_location

    def getLabelImage(self):
        return self.label_im