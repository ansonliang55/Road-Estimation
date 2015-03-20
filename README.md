# uot-ml-project
Autonomous Driving: Road-Estimation

Data file can be obtained from: 
http://www.cvlibs.net/datasets/kitti/eval_road.php 
(Download base kit with: left color images, calibration and training labels (0.5 GB))

Data split:
60% training, 10% validation, 30% testing

Purpose:
The purpose of this project is to investigate machine learning techniques to solve this task. Try things that we have seen in class and feel free to also try other techniques. Write in your report what you have done, what you observe, what you have tried, why you did what you did, etc.

Make a section of your report about what would you do to scale what you have done to work with pixels, as in that case you have millions of examples. You don’t need to implement anything but add a discussion of your ideas

Pystruct:
https://pystruct.github.io/auto_examples/image_segmentation.html#image-segmentation-py

Step:                                                                                  

1. slic pictures into super pixels (done)
1. feature extraction/preprocessing
  * "The unary potentials used in our implementation are derived from TextonBoost [19, 13]. We use
the 17-dimensional filter bank suggested by Shotton et al. [19], and follow Ladick´y et al. [13] by
adding color, histogram of oriented gradients (HOG), and pixel location features."
1. unary function derivation
2. add pairwise function. (Mar. 22th)
3. choose classification method.(random forest, neural network, SVM) (Mar. 31th)
4. implementation(python) (Apr. 3rd)
  * fully connected CRF     
  * max flow - min cut algorithm.
1. evaluation benchmark (Apr. 6th)
2. report (Apr. 8th)

REF:
* [13] L. Ladick´y, C. Russell, P. Kohli, and P. H. S. Torr. Associative hierarchical crfs for object class image
segmentation. In Proc. ICCV, 2009. 1, 5
* [19] J. Shotton, J. M. Winn, C. Rother, and A. Criminisi. Textonboost for image understanding: Multi-class
object recognition and segmentation by jointly modeling texture, layout, and context. IJCV, 81(1), 2009.
1, 3, 5, 6



