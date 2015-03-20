/*
Copyright (c) 2007-2010 ICG TU Graz

This file is part of CovSigmaPointsComp.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

//	Contact:	Stefan Kluckner, kluckner@icg.tugraz.at

#include "integralWrapper.h"
#include "math.h"

void mexFunction(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]) 
{
	if (n_in == 0) 
	{
		mexPrintf("List of available functions:\n");
		mexPrintf("Call: integralWrapper('function',..)\n");
		mexPrintf("Functions:\n");
		mexPrintf("1. Build IntegralStructure:   function = 'buildIntegralStructure'\n");
		mexPrintf("   Example:   obj = integralWrapper('buildIntegralStructure', data)\n");

		mexPrintf("2. Release IntegralStructure: function = 'releaseIntegralStructure'\n");
		mexPrintf("   Example:   integralWrapper('releaseIntegralStructure', obj )\n");

		mexPrintf("3. Get Information: function = 'getFromIntegralStructure'\n");
		mexPrintf("   Example:   integralWrapper('getFromIntegralStructure', 'mode', obj, ROI )\n");
		mexPrintf("   Modes: a) mode = 'sumIntImage': Summed Values within a ROI\n");
		mexPrintf("          b) mode = 'meanIntImage': Mean Values within a ROI\n");
		mexPrintf("          d) mode = 'meanAndVariance': Mean Values and Covariance Matrix within a ROI\n");
		mexPrintf("          e) mode = 'covarianceSigma': SigmaPoints within a ROI\n");
		mexPrintf("   Additional Parameters:\n     ");
		mexPrintf("          IntegralObject obj\n" );
		mexPrintf("          ROI: [ left1 upper1 sizeX1 sizeY1;\n");
		mexPrintf("              	left2 upper2 sizeX2 sizeY2;\n");
		mexPrintf("              	left3 upper3 sizeX3 sizeY3; ... ]\n");

		mexPrintf("4. Compute Distances: function = 'getDistancesFromIntegralStructure'\n");
		mexPrintf("   Example:   integralWrapper('getDistancesFromIntegralStructure', 'mode', obj, ROI, referenceVector )\n");
		mexPrintf("   Modes: a) mode = 'covarianceSigma'\n");

		mexPrintf("5. Compute Distances: function = 'getDistanceMapFromIntegralStructure'\n");
		mexPrintf("   Example:   integralWrapper('getDistanceMapFromIntegralStructure', 'mode', obj, [sizeX sizeY], referenceVector )\n");
		mexPrintf("   Modes: a) mode = 'covarianceSigma'\n");

		mexPrintf("6. Compute Distances: function = 'getReducedDistanceMapFromIntegralStructure'\n");
		mexPrintf("   Example:   integralWrapper('getReducedDistanceMapFromIntegralStructure', 'mode', obj, [sizeX sizeY], [ left upper right bottom], referenceVector )\n");
		mexPrintf("   Modes: a) mode = 'covarianceSigma'\n");

		return;
	}

	const char *funName = NULL;
	if(!mxIsChar(prhs[0]))
	{
		mexErrMsgTxt("First argument must be a string with the function to call!\n");
		return;
	}
	else
		funName = (char *)mxArrayToString(prhs[0]);
	
	if(!strcmp(funName,"buildIntegralStructure"))
		IntegralStructure(n_out, plhs, n_in, prhs);
	else if(!strcmp(funName,"getFromIntegralStructure"))
		GetFromIntegralStructure(n_out, plhs, n_in, prhs);
	else if(!strcmp(funName,"releaseIntegralStructure"))
		ReleaseIntegralStructure(n_out, plhs, n_in, prhs);
	else if(!strcmp(funName,"getDistanceMapFromIntegralStructure"))
		GetDistanceMapFromIntegralStructure(n_out, plhs, n_in, prhs);
	else if(!strcmp(funName,"getReducedDistanceMapFromIntegralStructure"))
		GetReducedDistanceMapFromIntegralStructure(n_out, plhs, n_in, prhs);
	else if(!strcmp(funName,"getDistancesFromIntegralStructure"))
		GetDistancesFromIntegralStructure(n_out, plhs, n_in, prhs);
	
	else
		mexErrMsgTxt("Unrecognized function name!\n");
}

void IntegralStructure(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]) 
{
	if ( n_in != 2 || n_out != 1  ) 
	{
		mexErrMsgTxt("Usage: OBJ = integralWrapper( 'integralStructure', data );\n");
		mexErrMsgTxt("       data - double MxNx3 rgb OR a double MxNx1 intensity image\n");
		mexErrMsgTxt("       Keep in mind to delete the object.\n");
		mexErrMsgTxt("       integralWrapper('releaseIntegralStructure', OBJ)\n");
		return;
	}

	IntegralStructures<t1,t2>* integralImage;
	
	const int* dim_array;
	dim_array = mxGetDimensions(prhs[1]);

	int dim = mxGetNumberOfDimensions(prhs[1]);
	int h = dim_array[0];
	int w = dim_array[1];

	int layers = 1;
	if ( dim > 2 )
		layers = dim_array[2];

	// mexPrintf("Build IntegralImage: dim: %d dim, layers: %d, width: %d, height: %d\n", dim, layers, w, h);

	// build external data structure
	t1* input_img = new t1[w * h * layers];

	// pointer to matlab data
	double * ptr_d;
	ptr_d = mxGetPr(prhs[1]);
		
	// assignment
	// slow !!!!
	// todo: pointer incr, memcpy
	for(int l = 0; l < layers; l++) 
	{
		for(int i = 0; i < w; i++) 
		{
			for(int j = 0; j < h; j++)
			{
				input_img[i+ j*w +l*w*h] = (t1)ptr_d[j];
			}
			ptr_d += h;
		}
	}

	integralImage = new IntegralStructures<t1,t2>( input_img,layers,h,w,1|2 );

	delete[] input_img;
		
	ObjectHandle<IntegralStructures<t1,t2> >* handle = 
			new ObjectHandle<IntegralStructures<t1,t2> >(integralImage);

	plhs[0] = handle->to_mex_handle();
}

void GetDistanceMapFromIntegralStructure( int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[] )
{
	if (n_in != 5) 
	{ 
		mexErrMsgTxt("Usage: integralWrapper( 'integralStructure', data );\n");
		return;
	}

	const char *modeName = 0;
	modeName = (char *)mxArrayToString( prhs[1] );

	ObjectHandle<IntegralStructures<t1,t2> >* handle = 
		ObjectHandle<IntegralStructures<t1,t2> >::from_mex_handle(prhs[2]);
	
	int dim = handle->get_object().GetDimension();
	int width = handle->get_object().GetNbOfCols();
	int height = handle->get_object().GetNbOfRows();
	
	double * ptr_d;
	ptr_d = mxGetPr( prhs[3] );

	int window[2] = { (int)ptr_d[0], (int)ptr_d[1] };

	//mexPrintf("Window Size: %d %d\n", window[0], window[1] );

	int nbCovariances = dim*dim;
	int nbSigmaPoints = dim*(2*dim+1);
	
	float *sigmaPointsRef = new float[ nbSigmaPoints ];

	ptr_d = mxGetPr( prhs[4] );
	for ( int i = 0; i < nbSigmaPoints; ++i )
		sigmaPointsRef[i] = (float)ptr_d[i];

	plhs[0] = mxCreateNumericMatrix( height, width, mxDOUBLE_CLASS, mxREAL );
	ptr_d = mxGetPr( plhs[0] );

	// fill with maxValues
	for ( int xy = 0; xy < width*height; ++xy )
		ptr_d[ xy ] = 1.0e11;

	// sigma points
	if ( !strcmp( modeName, "covarianceSigma") ) 
	{
		for ( int x = 0; x < (width-window[1]); ++x )
		{
			for ( int y = 0; y < (height-window[0]); ++y )
			{
				float *sigmaPoints = new float[ nbSigmaPoints ];
				
				myRect roi;
				roi.left = x;
				roi.upper = y;
				roi.height = window[0];
				roi.width = window[1];

				// mode
				bool valid = false;
				handle->get_object().GetSigmaPointFeature( roi, sigmaPoints, valid );

				if ( valid )
				{
					// compute euclidean distance
					float distance = 0.0;

					for ( int i = 0; i < dim; ++i )
						distance  += (sigmaPoints[i]-sigmaPointsRef[i])*(sigmaPoints[i]-sigmaPointsRef[i]);

					for ( int i = 0; i < dim*dim; ++i )
					{
						distance  += (sigmaPoints[dim + i]-sigmaPointsRef[dim + i])*(sigmaPoints[dim + i]-sigmaPointsRef[dim + i]);
						distance  += (sigmaPoints[dim*dim + dim + i]-sigmaPointsRef[dim*dim + dim + i])*
							(sigmaPoints[dim*dim + dim + i]-sigmaPointsRef[dim*dim + dim + i]);
					}

					ptr_d[ x*height + y] = sqrt( distance );
				}
				else
				{
					// set to max value else
					ptr_d[ x*height + y] = 1.0e11;
				}

				delete[] sigmaPoints;
			}
		}
	}
	else
	{
		// todo
		// sigma points without mean
		// ...
	}
	
	delete[] sigmaPointsRef;
}

void GetReducedDistanceMapFromIntegralStructure(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]) 
{
	if (n_in != 6) 
	{ 
		mexErrMsgTxt("Usage: integralWrapper( 'integralStructure', data );\n");
		return;
	}

	const char *modeName = 0;
	modeName = (char *)mxArrayToString( prhs[1] );

	ObjectHandle<IntegralStructures<t1,t2> >* handle = 
		ObjectHandle<IntegralStructures<t1,t2> >::from_mex_handle(prhs[2]);
	
	int dim = handle->get_object().GetDimension();
	int width = handle->get_object().GetNbOfCols();
	int height = handle->get_object().GetNbOfRows();
	
	double * ptr_d;
	ptr_d = mxGetPr( prhs[3] );

	int window[2] = { (int)ptr_d[0], (int)ptr_d[1] };


	ptr_d = mxGetPr(prhs[4]);
	int roiComp[4] = { (int)ptr_d[0], (int)ptr_d[1], (int)ptr_d[2], (int)ptr_d[3] };
	
	//mexPrintf("Window Size: %d %d\n", window[0], window[1] );
	//mexPrintf("Window Size: %d %d %d %d\n", roiComp[0], roiComp[1], roiComp[2], roiComp[3] );

	int nbCovariances = dim*dim;
	int nbSigmaPoints = dim*(2*dim+1);
	
	float *sigmaPoints = new float[ nbSigmaPoints ];
	float *sigmaPointsRef = new float[ nbSigmaPoints ];

	ptr_d = mxGetPr( prhs[5] );
	for ( int i = 0; i < nbSigmaPoints; ++i )
		sigmaPointsRef[i] = (float)ptr_d[i];

	plhs[0] = mxCreateNumericMatrix( height, width, mxDOUBLE_CLASS, mxREAL );
	ptr_d = mxGetPr( plhs[0] );

	// fill with maxValues
	for ( int xy = 0; xy < width*height; ++xy )
		ptr_d[ xy ] = 1.0e11;

	if ( !strcmp(modeName ,"covarianceSigma") ) 
	{
		for ( int x = roiComp[1]; x < std::min( width-window[1], roiComp[3] ); ++x )
		{
			for ( int y = roiComp[0]; y < std::min( height-window[0], roiComp[2] ); ++y )
			{
				myRect roi;
				roi.height = window[0];
				roi.width = window[1];
				roi.left = x;
				roi.upper = y;
	
				float *sigmaPoints = new float[ nbSigmaPoints ];
				
				// mode
				bool valid = false;
				handle->get_object().GetSigmaPointFeature( roi, sigmaPoints, valid );

				if ( valid )
				{
					float distance = 0.0;

					for ( int i = 0; i < dim; ++i )
						distance  += (sigmaPoints[i]-sigmaPointsRef[i])*(sigmaPoints[i]-sigmaPointsRef[i]);

					for ( int i = 0; i < dim*dim; ++i )
					{
						distance  += (sigmaPoints[dim + i]-sigmaPointsRef[dim + i])*(sigmaPoints[dim + i]-sigmaPointsRef[dim + i]);
						distance  += (sigmaPoints[dim*dim + dim + i]-sigmaPointsRef[dim*dim + dim + i])*
							(sigmaPoints[dim*dim + dim + i]-sigmaPointsRef[dim*dim + dim + i]);
					}

					ptr_d[ x*height + y] = sqrt( distance  );
				}
				else
				{
					// set to max value else
					ptr_d[ x*height + y] = 1.0e11;
				}

				delete[] sigmaPoints;
			}
		}
	}
	else 
	{
		// todo
		// sigma points without mean
		// ...
	}

	delete[] sigmaPointsRef;
}




void GetDistancesFromIntegralStructure(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]) 
{
	
	if ( n_in != 5 ) 
	{ 
		mexErrMsgTxt("Usage: integralWrapper( 'integralStructure', data );\n");
		return;
	}

	const char *modeName = 0;
	modeName = (char *)mxArrayToString( prhs[1] );

	ObjectHandle<IntegralStructures<t1,t2> >* handle = 
		ObjectHandle<IntegralStructures<t1,t2> >::from_mex_handle(prhs[2]);

	double * ptr_d, *ptrRoi, *ptrDist;

	const int *dim_array;
	dim_array = mxGetDimensions(prhs[3]);
	int nbRoi= dim_array[0];
	int rects = dim_array[1];

	//mexPrintf("ROIs: %d %d\n", nbRoi, rects );
	
	int dim = handle->get_object().GetDimension();
	int nbCovariances = dim*dim;
	int nbSigmaPoints = dim*(2*dim+1);
	
	float *sigmaPoints = new float[ nbSigmaPoints ];
	float *sigmaPointsRef = new float[ nbSigmaPoints ];

	plhs[0] = mxCreateNumericMatrix( 1, nbRoi, mxDOUBLE_CLASS, mxREAL );

	ptrDist = mxGetPr( plhs[0] );
	ptrRoi = mxGetPr( prhs[3] );
	ptr_d = mxGetPr( prhs[4] );

	for ( int i = 0; i < dim*(2*dim+1); ++i )
		sigmaPointsRef[i] = (float)ptr_d[i];

	if ( !strcmp(modeName ,"covarianceSigma") ) 
	{
		for ( int i = 0; i < nbRoi; ++i )
		{
			myRect roi( (int)ptrRoi[0], (int)ptrRoi[nbRoi], (int)ptrRoi[2*nbRoi], (int)ptrRoi[3*nbRoi] );

			float *sigmaPoints = new float[ nbSigmaPoints ];
			memset( sigmaPoints, 0x00, sizeof(float)*nbSigmaPoints );
				
				// mode
			bool valid = false;
			handle->get_object().GetSigmaPointFeature( roi, sigmaPoints, valid );

			if ( valid )
			{
				float distance = 0.0;

				for ( int i = 0; i < dim; ++i )
					distance  += (sigmaPoints[i]-sigmaPointsRef[i])*(sigmaPoints[i]-sigmaPointsRef[i]);

				for ( int i = 0; i < dim*dim; ++i )
				{
					distance  += (sigmaPoints[dim + i]-sigmaPointsRef[dim + i])*(sigmaPoints[dim + i]-sigmaPointsRef[dim + i]);
					distance  += (sigmaPoints[dim*dim + dim + i]-sigmaPointsRef[dim*dim + dim + i])*
						(sigmaPoints[dim*dim + dim + i]-sigmaPointsRef[dim*dim + dim + i]);
				}

				*ptrDist = sqrt( distance );
			}
			else
			{
				*ptrDist = 1.0e11;
			}

			delete[] sigmaPoints;

			ptrRoi++;
			ptrDist++;
			
		}
	}
	else 
	{
		// todo
	}

	delete[] sigmaPointsRef;
}

void GetFromIntegralStructure(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]) 
{
	if ( n_in != 4 ) 
	{ 
		mexErrMsgTxt("Usage: integralWrapper( 'integralStructure', data );\n");
		return;
	}

	const char *modeName = 0;
	modeName = (char *)mxArrayToString(prhs[1]);
	
	double * ptr_d = 0;
	double * ptrRoi = 0;


	ObjectHandle<IntegralStructures<t1,t2> >* handle = 
		ObjectHandle<IntegralStructures<t1,t2> >::from_mex_handle(prhs[2]);
	
	int dim = handle->get_object().GetDimension();

	const int *dim_array;
	dim_array = mxGetDimensions(prhs[3]);
	
	int nbRoi= dim_array[0];
	int rects = dim_array[1];
	ptrRoi = mxGetPr( prhs[3] );

	//mexPrintf("Rect: %d %d %d %d\n",(int)ptr_d[0], (int)ptr_d[1], (int)ptr_d[2], (int)ptr_d[3]);

	if (!strcmp( modeName, "sumIntImage")) 
	{
		plhs[0] = mxCreateNumericMatrix( nbRoi, dim, mxDOUBLE_CLASS, mxREAL);
		ptr_d = mxGetPr(plhs[0]);

		for ( int j = 0; j < nbRoi; ++j )
		{
			float *sum = new float[dim];
			myRect roi( (int)ptrRoi[0], (int)ptrRoi[nbRoi], (int)ptrRoi[2*nbRoi], (int)ptrRoi[3*nbRoi] );
			handle->get_object().GetSumIntImage(roi, sum);

			for( int i = 0; i < dim; i++)
				ptr_d[i*nbRoi + j] = sum[i];

			ptrRoi++;
			delete[] sum;
		}
	}
	else
	if(!strcmp(modeName, "meanIntImage")) 
	{
		plhs[0] = mxCreateNumericMatrix(  nbRoi, dim,mxDOUBLE_CLASS, mxREAL);
		ptr_d = mxGetPr(plhs[0]);

		for ( int j = 0; j < nbRoi; ++j )
		{
			bool valid;
			float *mean = new float[dim];
			myRect roi( (int)ptrRoi[0], (int)ptrRoi[nbRoi], (int)ptrRoi[2*nbRoi], (int)ptrRoi[3*nbRoi] );
			handle->get_object().GetMeanIntImage( roi, mean, valid );

			for( int i = 0; i < dim; i++) 
				ptr_d[i*nbRoi + j] = mean[i];

			ptrRoi++;
			delete[] mean;
		}
	}
	else
	if(!strcmp(modeName, "meanAndVariance")) 
	{
		int nbCovariances = dim*dim;
		plhs[0] = mxCreateNumericMatrix(  nbRoi, dim + nbCovariances, mxDOUBLE_CLASS, mxREAL);
		ptr_d = mxGetPr(plhs[0]);

		for ( int j = 0; j < nbRoi; ++j )
		{
			float *mean = new float[dim];
			float *variance = new float[nbCovariances];
			myRect roi( (int)ptrRoi[0], (int)ptrRoi[nbRoi], (int)ptrRoi[2*nbRoi], (int)ptrRoi[3*nbRoi] );
			handle->get_object().GetMeanAndVariance(roi, mean, variance);
	
			for( int i = 0; i < dim; i++)
				ptr_d[i*nbRoi + j] = mean[i];
				
			for( int i = dim; i < dim+nbCovariances; i++)
				ptr_d[i*nbRoi + j] = variance[i-dim];

			ptrRoi++;
			delete[] mean;
			delete[] variance;
		}
	}
	else 
	if (!strcmp(modeName, "covarianceSigma") )  
	{
		int nbFeat = (2*dim+1)*dim;

		plhs[0] = mxCreateNumericMatrix(  nbRoi, nbFeat, mxDOUBLE_CLASS, mxREAL );
		ptr_d = mxGetPr(plhs[0]);

		for ( int j = 0; j < nbRoi; ++j )
		{
			float *sigmaPoints = new float[(2*dim+1)*dim];
			myRect roi( (int)ptrRoi[0], (int)ptrRoi[nbRoi], (int)ptrRoi[2*nbRoi], (int)ptrRoi[3*nbRoi] );
			
			bool valid = false;
			handle->get_object().GetSigmaPointFeature( roi, sigmaPoints, valid );

			for( int i = 0; i < nbFeat; ++i )
			{
				ptr_d[ i*nbRoi + j] = (double)sigmaPoints[i];
			}

			delete[] sigmaPoints;
			ptrRoi++;
		}
	}
}

void ReleaseIntegralStructure(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]) 
{
	if ( n_in != 2 ) 
	{ 
		mexErrMsgTxt("Usage: integralWrapper('releaseIntegralStructure', OBJ);\n");
		return;
	}

	ObjectHandle<IntegralStructures<t1,t2> >* handle =
		ObjectHandle<IntegralStructures<t1,t2> >::from_mex_handle(prhs[1]);
	
	delete handle;
}






