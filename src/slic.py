"""
Written by: Anson Liang
"""

from skimage.segmentation import slic
from superpixel import showPlots


# @input 
#      image: 3D array nxmx3 pixels with 3 colors. 
#      numSegments: number of desired segments  
#      com_factor: the compact factors
# @output
#      segments: 2D array nxm describing pixel label
def getSlicSuperpixels(image, numSegments, com_factor):
    superpixels = slic(image, n_segments = numSegments, sigma = com_factor)
    # code for display results
    #showPlots(image, numSegments, superpixels)
    return superpixels



""" get color histogram code
def get_color_histogram(image,superpixels,index):
    indices = np.where(grid.ravel() == index)[0]
    r = np.bincount(im[:,:,0].ravel()[indices],minlength=256)
    g = np.bincount(im[:,:,1].ravel()[indices],minlength=256)
    b = np.bincount(im[:,:,2].ravel()[indices],minlength=256)
  
    histogram = (rhist0+bhist0+ghist0)/3
    return histogram
  """
