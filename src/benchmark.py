"""
Written by: Wenjie Zi

This is the handler of benchmark
"""
import numpy as np
#@input: start time and end time
#@output: the last of the time

def countTime(startTime, endTime):
    lastTime = endTime - startTime
    print ('Time used: %f')%lastTime
    return lastTime

#@input: 2D result of the prediction
#        2D labels
#@output: the probability of accuracy in superpixel level

def accuracyOfSuperpixels(file_num,valid_files, valid_data, clf, valid_labels,validationOriginalImage):
    count_correct=0
 #   print valid_files.shape
    indices = np.where(valid_files == file_num)[0]
    indices = np.array(indices)
    total_samples = indices.shape[0]
    for i in range(0,total_samples):
        if clf.predict(valid_data[indices[i]]) == valid_labels[indices[i]]:
                count_correct+=1
    print validationOriginalImage[file_num]
    print ('Validation Accuracy (Superpixel level): %2.2f%%')%(100.0*count_correct/total_samples)
<<<<<<< HEAD
    return 100.0*count_correct/total_samples

=======
    return count_correct, total_samples
>>>>>>> test
#@input: 2D result of the prediction
#        2D labels
#@output: the probility of accuracy in pixel level
def accuracyOfPixels(file_num,valid_files, superpixels, valid_data, clf, valid_pixels_labels,validationOriginalImage):
    count_correct = 0
    total_count = 0

    indicess = np.where(valid_files == file_num)[0]
    total_samples = indicess.shape[0]
    for i in range(0,total_samples):
        temp = np.array(superpixels[file_num][0])
<<<<<<< HEAD
 
        temp2 = np.array(valid_pixels_labels[file_num][0])
        index = np.array(np.where(temp == i)).T

=======
        temp2 = np.array(valid_pixels_labels[file_num][0])
        index = np.array(np.where(temp == i)).T
>>>>>>> test
        total_count = total_count + index.shape[0]
        predict_result = clf.predict(valid_data[indicess[i]])
        #print predict_result == temp2[index[0:index.shape[]]]
        for j in range(0,index.shape[0]): 
            if predict_result == temp2[index[j][0]][index[j][1]]:
               count_correct+=1
<<<<<<< HEAD

    print ('Validation Accuracy (Pixel level): %2.2f%%')%(100.0*count_correct/total_count)
=======
    print validationOriginalImage[file_num]
    print ('Validation Accuracy (Pixel level): %2.2f%%')%(100.0*count_correct/total_count)
    return count_correct, total_count

def overrallAverageResult(superpixelCorrect, superpixelTotal,pixelCorrect, pixelTotal):
    print ('The Validation Set Accuracy (SuperPixel level): %2.2f%%')%(100.0*superpixelCorrect/superpixelTotal)
    print ('The Validation Set Accuracy (Pixel level): %2.2f%%')%(100.0*pixelCorrect/pixelTotal)
>>>>>>> test
