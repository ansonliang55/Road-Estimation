

DISCLAIMER:
Please use it at your own risk. The executable is provided only for the purpose of evalualtion of the algorithm presented in the paper "SLIC Superpixels". Neither the authors of the paper nor EPFL can be held responsible for any damages resulting from use of this software.

USAGE:

--------------------------------------------------------------------------------------------
./slicSuperpixels_ubuntu64bits image_name <number_of_superpixels> <spatial_proximity_weight>
--------------------------------------------------------------------------------------------

The default <number_of_superpixels> is 200 and the default <spatial_proximity_weight> is 10.

The former value can range from 2 to the number of pixels in the image. The latter can vary from 1 to 30 depending on the application requirements.

As output, the algorithm generates two files:
- the segmented file showing boundaries overlaid on the original image
- a filename.dat file which contains labels of the segments.

INFO ON filename.data
---------------------

This is a binary file containing the label of each pixel (as a 32 bit integer value) in raster scan order with no other characters (like spaces or commas).

They can be read in C using the following function (or something similar):

	//------------------------------------------------------
	FILE* pf = fopen("C:/rktemp/filename.dat", "r");
	int sz = width*height;
	int* vals = new int[sz];
	int elread = fread((char*)vals, sizeof(int), sz, pf);
	for( int j = 0; j < height; j++ )
	{
		for( int k = 0; k < width; k++ )
		{
			int i = j*width+k;
			labels[i] = vals[i];
		}
	}
	delete [] vals;
	fclose(pf);
	//------------------------------------------------------


In this function, width and height of the image are provided as input. labels is an array that contains the labels in raster scan order.




