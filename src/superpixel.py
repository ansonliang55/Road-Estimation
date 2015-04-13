"""
Written by: Anson Liang, Wenjie Zi

This is a handler for superpixels
"""
import numpy as np
from skimage.segmentation import mark_boundaries
from skimage.feature import hog
import matplotlib.pyplot as plt
from sklearn import preprocessing
from skimage import exposure

def getSuperValidFiles(superpixels, count, valid_files):
    numSuperpixels = np.max(superpixels)+1
    for i in xrange(0,numSuperpixels):
        valid_files.append(count)
    return valid_files

# @input 
#      superpixels: 2D array nxm pixels label 
# @output
#      locations: 2D array nx2 centroid locaiton of all superpixels
def getSuperPixelLocations(superpixels):
    locations = []
    numSuperpixels = np.max(superpixels)+1
    for i in xrange(0,numSuperpixels):
        indices = np.where(superpixels == i)
        x = np.mean(indices[0])
        y = np.mean(indices[1])
        locations.append([x,y])
    return np.array(locations)

# @input 
#      superpixels: 2D array nxm pixels label 
# @output
#      locations: 2D array 64xK dimension shape mask of all superpixels
def getSuperPixelShape(superpixels):

    shape = []
    numSuperpixels = np.max(superpixels)+1
    for i in xrange(0,numSuperpixels):
        temp = np.zeros((1,64),dtype = float)
        indices = np.where(superpixels == i)
        width = np.max(indices[0]) - np.min(indices[0])
        height = np.max(indices[1]) - np.min(indices[1])
        boxWidth = np.max((width, height))
        boundingBox = np.zeros((boxWidth+1, boxWidth+1))
        x = indices[0] - np.min(indices[0])
        y = indices[1] - np.min(indices[1])
        boundingBox[(x,y)] = 1
        boundingBox = boundingBox.ravel()
        binWidth = 1.0*len(boundingBox)/64
        for j in xrange(0,len(boundingBox)):
            if boundingBox[j] == 1:
                x = j/binWidth
                temp[0][x] = temp[0][x]+1
        shape.append(1.0*temp[0]/np.sum(temp[0]))
    return np.array(shape)

# @input 
#      superpixels: 2D array nxm pixels label 
#      image: 3D array nxmx3 pixels color original image
# @output
#      color: 2D array nx3 mean color of all superpixels
def getSuperPixelMeanColor(superpixels, image):
    colors = []
    #newIm = image
    numSuperpixels = np.max(superpixels)+1
    for i in xrange(0,numSuperpixels):
        indices = np.where(superpixels==i)
        color = image[indices]
        r=np.mean(color.T[0])
        g=np.mean(color.T[1])
        b=np.mean(color.T[2])
        #newIm[indices] = [r,g,b]
        colors.append([r,g,b])
    #showPlots(newIm, numSuperpixels, superpixels)
    return np.array(colors)

# @input 
#      superpixels: 2D array nxm pixels label 
#      image: 3D array nxmx3 pixels color original image
# @output
#      color: 64D array color histogram of all superpixels
def getSuperPixelColorHistogram(superpixels, image):
    colors = []
    #newIm = image
    numSuperpixels = np.max(superpixels)+1
    for i in xrange(0,numSuperpixels):
        temp = np.zeros((1,64),dtype = float)
        indices = np.where(superpixels==i)
        color = image[indices]
        for j in xrange(0,color.shape[0]):
            r = np.int_(color[j][0]/0.25)
            g = np.int_(color[j][1]/0.25)
            b = np.int_(color[j][2]/0.25)
            if r ==4:
                r = 3
            if g == 4:
                g = 3
            if b == 4:
                b = 3
            x = 16*r+4*g+b*1
            temp[0][x] = temp[0][x]+1
        #min_max_scaler = preprocessing.MinMaxScaler()
        #t = min_max_scaler.fit_transform(temp[0])
        #print t
        colors.append(temp[0])
    #showPlots(newIm, numSuperpixels, superpixels)
    return np.array(colors)

# @input 
#      superpixels: 2D array nxm pixels label 
#      label_image: 3D array nxmx3 pixels color original image
#      thres: ratio threshold (0-1) for setting a superpixel to be true
# @output
#      gradient: histogram of oriented gradient
def getSuperPixelOrientedHistogram(superpixels,image):
    gradients = []
    numSuperpixels = np.max(superpixels)+1
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    for i in xrange(0,numSuperpixels):
        temp = np.zeros((1,10),dtype = int)
        indices = np.where(superpixels==i)
        gradient = hog_image_rescaled[indices]
        for j in xrange(0, gradient.shape[0]):
            x = np.int_(gradient[j] / 0.1)
            if x == 10:
                x = 9
            temp[0][x] = temp[0][x] + 1
        #preprocessing.normalize(temp[0])
        gradients.append(temp[0])

    #showPlots(newIm, numSuperpixels, superpixels)
    return np.array(gradients)

# @input 
#      label_image: 3D array nxmx3 pixels color original image
# @output
#      label_pixels: pixel level labels
def getPixelLabel(label_image):
    labelValue = label_image.max()
    label_pixels = (label_image == labelValue)
    return np.array(label_pixels)

# @input 
#      superpixels: 2D array nxm pixels label 
#      label_image: 3D array nxmx3 pixels color original image
#      thres: ratio threshold (0-1) for setting a superpixel to be true
# @output
#      superpixel_labels: 1D Array label for super pixel (only 1 or 0)
def getSuperPixelLabel(superpixels, label_image, thres):
    #newIm = image
    superpixel_labels = []
    numSuperpixels = np.max(superpixels)+1
    labelValue = label_image.max()
    label_pixels = (label_image == labelValue)
    for i in xrange(0,numSuperpixels):
        indices = np.where(superpixels==i)
        cor_label = label_pixels[indices]
        portion_true = 1.0*np.sum(cor_label)/len(cor_label)
        if portion_true > thres:
            superpixel_labels.append(1)
            #newIm[indices] = [1,1,1]
        else:
            superpixel_labels.append(0)
            #newIm[indices] = [0,0,0]
    #showPlots(newIm, numSuperpixels, superpixels)
    # show sample test mean image
    return np.array(superpixel_labels)

# @input 
#      superpixels: 2D array nxm pixels label 
#      label_image: 3D array nxm pixels color original image
# @output
#      superpixel_labels: 1D Array label for super pixel (between 0-1)
def getSuperPixelLabelPercent(superpixels, label_image):
    #newIm = image
    superpixel_labels = []
    numSuperpixels = np.max(superpixels)+1
    labelValue = label_image.max()
    label_pixels = (label_image == labelValue)
    for i in xrange(0,numSuperpixels):
        indices = np.where(superpixels==i)
        cor_label = label_pixels[indices]
        portion_true = 1.0*np.sum(cor_label)/len(cor_label)
        superpixel_labels.append(portion_true)
    #showPlots(newIm, numSuperpixels, superpixels)
    #show sample test mean image
    return np.array(superpixel_labels)

# @input 
#      superpixels: 2D array nxm pixels label 
# @output
#      superpixel_size: 1D array number of pixels per superpixels
def getSuperPixelSize(superpixels):
    superpixel_size = []
    numSuperpixels = np.max(superpixels)+1
    for i in xrange(0,numSuperpixels):
        size = np.sum(superpixels==i)
        superpixel_size.append(size)
    #preprocessing.normalize(superpixel_size)
    return np.array(superpixel_size)


# @input 
#      im_name: name the image output
#      superpixels: 2D array nxm pixels label 
#      image: 3D array nxmx3 pixels color original image
#      numSuperpixels: number of superpixels in superpixels
# @output (visualize data)
#      just display the image with marked superpixel boundary
def showPlots(im_name, image, numSuperpixels, superpixels):
    # show sample test mean image
    fig = plt.figure("Superpixels -- %d im_sp" % (numSuperpixels))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, superpixels))
    #bugged
    #ax.imsave("output/000%d.png" % (im_name),mark_boundaries(image, superpixels))
    plt.axis("off")
    plt.show()

# @input 
#      clf: classficiation object containing trained parameters
#      superpixels: 2D array nxm pixels label 
#      test_data: 2D array nxm, n data, m features
#      image: 2D array nxm original image
# @output
#      display output image indicating road
def showPrediction(im_name, clf, superpixels, test_data, image):
    newIm = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
    numSuperpixels = np.max(superpixels)+1
    for i in xrange(0,numSuperpixels):
        indices = np.where(superpixels==i)
        prediction = clf.predict_proba([test_data[i]])[0][1]
        #if prediction == 1:
        newIm[indices] = [0,prediction,0]
        #else:
        #    newIm[indices] = [0,0,0]
        indices = np.where(newIm > 0.5)
        
        image[indices] = newIm[indices]
    showPlots(im_name, image, numSuperpixels, superpixels)

# @input 
#      superpixels: 2D array nxm pixels label 
# @output
#      edges: pairwise matrix   
def getPairwiseMatrix(superpixels):
    numSuperpixels = np.max(superpixels)+1
    [row, col] = superpixels.shape
    edges = np.zeros((numSuperpixels,numSuperpixels))
    for i in xrange(0, row-1):
        for j in xrange(0, col-1):
            if(superpixels[i][j] != superpixels[i][j+1]):
                edges[superpixels[i][j]][superpixels[i][j+1]] = 1
                edges[superpixels[i][j+1]][superpixels[i][j]] = 1
            if(superpixels[i][j] != superpixels[i+1][j]):
                edges[superpixels[i][j]][superpixels[i+1][j]] = 1
                edges[superpixels[i+1][j]][superpixels[i][j]] = 1
    return edges


