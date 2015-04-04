"""
Written by: Wenjie Zi

This is the handler of benchmark
"""
import numpy as np
#@input: start time and end time
#@output: the last of the time

def countTime(startTime, endTime):
    lastTime = startTime, endTime
    print ('Time used: %2.2f%%')%lastTime
    return lastTime

#@input: 2D result of the prediction
#        2D labels
#@output: the probility of accuracy in superpixel level

def accuracyOfSuperpixels(valid_data, clf, valid_labels):
	count_correct=0
	total_sample = len(valid_data)
	for i in xrange(0,total_sample):
		if clf.predict(valid_data[i]) == valid_labels[i]:
				count_correct+=1

	print ('Validation Accuracy (Superpixel level): %2.2f%%')%100.0*count_correct/total_sample
	return 100.0*count_correct/total_sample
#@input: 2D result of the prediction
#        2D labels
#@output: the probility of accuracy in pixel level
def accuracyOfPixels(superpixels, valid_data, clf, valid_pixels_labels):
	count_correct=0
	total_sample = valid_pixels_labels.shape[0]*valid_pixels_labels.shape[1]
	total_superpixles = len(valid_data)
	for i in xrange(0,total_superpixles):
		indices = np.where(superpixels == i)
		index = valid_pixles_labels[indices]
	    predictResult = clf.predict(valid_data[i])
		for j in xrange(0,index.shape):	
			if predictResult == index[j]
			   count_correct+=1
	print ('Validation Accuracy (Pixel level): %2.2f%%')%100.0*count_correct/total_sample
	return 100.0*count_correct/total_sample

def 