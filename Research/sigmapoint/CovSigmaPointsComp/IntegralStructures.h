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

#define _INTSTRUCTURE_H_
#include <string.h>
#include <math.h>

// struct for rectangular region
struct myRect
{
	public:

	/** Rectangle Constructor */
	myRect()
	{
		this->upper = -1;
		this->left = -1;
		this->height = -1;
		this->width = -1;
	}

	~myRect()
	{
	}

	/** Rectangle Constructor */
	myRect( int upper, int left, int height, int width )
	{
		this->upper = upper;
		this->left = left;
		this->height = height;
		this->width = width;
	}

	/** Defines the Upper Value */
	int upper;

	/** Defines the Left Value */
	int left;

	/** Defines the Height Value */
	int height;

	/** Defines the Width Value */
	int width; 
};

/**
 *  \class IntegralStructures
 *  \brief This class handles the Integral Structures
 *  \ingroup InputOutput
 */
template< typename INPUTDATATYPE, typename INTERNALDATATYPE >
class IntegralStructures
{
public:
	/** \enum Integral Structures */
	
	// Binary, Integer with 32 Bit
	// 1: 00 ... 00001 Image
	// 2: 00 ... 00010 Squared Image, Covariance Computation
	enum IntegralImages
	{   
		INT_IMAGE = 1, /**< Image Representation */ 
		INT_SQIMAGE = 2, /**< Image Squared Representation */
	};

	/**
	 * Constructor
	 * \brief Default constructor
	 * \brief Format Input Image, row - wise
     * \brief x1 .. layer 1, x2 .. layer 2, ..
	 * \brief [x11 x12 x13 ... , x21 x22 x23 .. , x31 x32 x33 .. ] 
	 *  
	 */ 
	IntegralStructures( INPUTDATATYPE *inputImage, 
		                const unsigned int dimension,
						const unsigned int nbRows, 
						const unsigned int nbCols,
						const int intTypes )
	{
		m_Dimension = dimension;
		m_NbElementsPerDimension = nbRows*nbCols;
		m_NbExtendedElementsPerDimension = (nbRows+1)*(nbCols+1);

		m_NbCovariances = ( m_Dimension*(m_Dimension+1) ) / 2;
		
		m_NbRows = nbRows;
		m_NbCols = nbCols;

		m_NbColsExtended = nbCols + 1;

		m_IntegralTypes = intTypes;

		m_TmpSum = new float[m_Dimension];
		m_TmpSquaredSum = new float[m_NbCovariances];

		m_Assignment = new unsigned int[4*m_NbCovariances];

		m_IntegralSquaredImage = 0;
		m_IntegralImage= 0;

		SetNewImage( inputImage );
	}


	/**
	 * Constructor
	 * \brief Default destructor
	 */
	~IntegralStructures()
	{
		if ( m_IntegralTypes & INT_IMAGE )
		{ 
			delete[] m_IntegralImage;
		}

		if ( m_IntegralTypes & INT_SQIMAGE )
		{
			delete[] m_IntegralSquaredImage;
		}

		delete[] m_TmpSum;
		delete[] m_TmpSquaredSum;
		delete[] m_Assignment;
	}

	/**
	 * \brief Set new image data with same dimension
	 */
	void SetNewImage( INPUTDATATYPE *inputImage )
	{
		if ( m_IntegralTypes & INT_IMAGE )
		{ 
			if ( m_IntegralImage != 0 )
			{
				delete[] m_IntegralImage;
			}

			BuildIntegralImage( inputImage );
		}

		if ( m_IntegralTypes & INT_SQIMAGE )
		{
			if ( m_IntegralSquaredImage != 0 )
			{
				delete[] m_IntegralSquaredImage;
			}

			BuildIntegralSquaredImage( inputImage );
		}
	}


	/**
	 * \brief Returns the squared summed values in a given roi
	 */
	void GetSquaredSumIntImage( const myRect& roi, float* sum )
	{
		unsigned int pos1 = roi.width;
		unsigned int pos2 = m_NbColsExtended*(roi.height); 
		unsigned int pos3 = pos1 + pos2;
		unsigned int offset = roi.left + m_NbColsExtended*roi.upper;

		for ( unsigned int currentDim = 0; currentDim < m_NbCovariances; ++currentDim )
		{
			const INTERNALDATATYPE *pt = &m_IntegralSquaredImage[ offset + 
				currentDim*m_NbExtendedElementsPerDimension ];

			INTERNALDATATYPE sumTmp = pt[0] + pt[pos3] - pt[ pos1 ] - pt[ pos2 ];	

			sum[currentDim] = (float)sumTmp;
		}
	}

	/**
	 * \brief Returns the summed values in a given roi
	 */
	void GetSumIntImage( const myRect& roi, float *sum )
	{
		unsigned int pos1 = roi.width;
		unsigned int pos2 = m_NbColsExtended*(roi.height); 
		unsigned int pos3 = pos1 + pos2;
		unsigned int offset = roi.left + m_NbColsExtended*roi.upper;

		for ( unsigned int currentDim = 0; currentDim < m_Dimension; ++currentDim )
		{
			const INTERNALDATATYPE *pt = &m_IntegralImage[ offset + 
				currentDim*m_NbExtendedElementsPerDimension  ];
			
			INTERNALDATATYPE sumTmp = pt[0] + pt[pos3] - pt[ pos1 ] - pt[ pos2 ];	

			sum[currentDim] = (float)sumTmp;
		}
	}

	/**
	 * \brief Returns the mean of a given roi
	 */
	void GetMeanIntImage( const myRect& roi,  float* mean, bool& valid )
	{
		GetSumIntImage( roi, m_TmpSum );
		float s = 1.0f/(roi.width*roi.height);
		for ( unsigned int currentDim = 0; currentDim < m_Dimension; ++currentDim )
		{
			mean[currentDim] = m_TmpSum[currentDim]*s;
		}
	}

	/**
	 * \brief Returns the variance values in a given roi
	 */
	void GetVarianceIntImage( const myRect& roi,  float* variance )
	{
		// approximation
		float area1 = 1.0f/( roi.width*roi.height-1.0f );
		float area2 = 1.0f/( roi.width*roi.height );

		GetSumIntImage( roi,  m_TmpSum );
		GetSquaredSumIntImage( roi, m_TmpSquaredSum );

		unsigned int count = 0;
		for ( unsigned int currentDim = 0; currentDim < m_NbCovariances; ++currentDim )
		{
			variance[ m_Assignment[count] ] = 
				variance[ m_Assignment[count + 1] ] = 
				area1*( m_TmpSquaredSum[currentDim] - 
				area2*m_TmpSum[ m_Assignment[count + 2]]*m_TmpSum[m_Assignment[count + 3]] );
			
			count += 4;
		}
	}

	/**
	 * \brief Returns the variance and mean values in a given roi in one step
	 */
	void GetMeanAndVariance( const myRect& roi, float* mean, float* variance )
	{
		int s = roi.width*roi.height;
		float area1 = 1.0f/( s-1.0f );
		float area2 = 1.0f/s;

		unsigned int pos1 = roi.width;
		unsigned int pos2 = m_NbColsExtended*roi.height; 
		unsigned int pos3 = pos1 + pos2;
		unsigned int offset = roi.left + m_NbColsExtended*roi.upper;

		for ( unsigned int currentDim = 0; currentDim < m_Dimension; ++currentDim )
		{
			const INTERNALDATATYPE *pt = &m_IntegralImage[ offset + 
				currentDim*m_NbExtendedElementsPerDimension ];

			INTERNALDATATYPE sum = pt[0] + pt[pos3] - pt[ pos1 ] - pt[ pos2 ];	

			m_TmpSum[currentDim] =	(float)sum;
			mean[currentDim] = m_TmpSum[currentDim]*area2;
		}

		for ( unsigned int currentDim = 0; currentDim < m_NbCovariances; ++currentDim )
		{
			const INTERNALDATATYPE *pt = &m_IntegralSquaredImage[ offset + 
				currentDim*m_NbExtendedElementsPerDimension ];
			
			INTERNALDATATYPE sum = pt[0] + pt[pos3] - pt[ pos1 ] - pt[ pos2 ];	

			m_TmpSquaredSum[currentDim] = static_cast<float>( sum );
		}

		unsigned int count = 0;
		for ( unsigned int currentDim = 0; currentDim < m_NbCovariances; ++currentDim )
		{
			variance[ m_Assignment[count] ] = variance[ m_Assignment[count + 1] ] = 
				area1*( m_TmpSquaredSum[currentDim] - 
				mean[ m_Assignment[count + 2] ]*m_TmpSum[ m_Assignment[count + 3] ] );

			count += 4;
		}
	}

	/**
	 * \brief Returns the sigma points in a given roi in one step
	 * sigmapoint vector size: DIM*(2*DIM+1)
	 */
	void GetSigmaPointFeature( const myRect& roi, float* sigmaPoints, bool& valid )
	{
		valid = true;
		// first extract mean and variance
		// *******************************
		int s = roi.width*roi.height;
		float area1 = 1.0f/( s-1.0f );
		float area2 = 1.0f/( s );

		unsigned int pos1 = roi.width;
		unsigned int pos2 = m_NbColsExtended*roi.height; 
		unsigned int pos3 = pos1 + pos2;
		unsigned int offset = roi.left + m_NbColsExtended*roi.upper;

		float* mean = new float[m_Dimension];
		float* variance = new float[m_Dimension*m_Dimension];

		for ( unsigned int currentDim = 0; currentDim < m_Dimension; ++currentDim )
		{
			const INTERNALDATATYPE *pt = &m_IntegralImage[ offset + 
				currentDim*m_NbExtendedElementsPerDimension];

			INTERNALDATATYPE sum = pt[0] + pt[pos3] - pt[ pos1 ] - pt[ pos2 ];	

			m_TmpSum[currentDim] = (float)sum;
			mean[currentDim] = m_TmpSum[currentDim]*area2;
		}

		for ( unsigned int currentDim = 0; currentDim < m_NbCovariances; ++currentDim )
		{
			const INTERNALDATATYPE *pt = &m_IntegralSquaredImage[ offset + 
				currentDim*m_NbExtendedElementsPerDimension];
			
			INTERNALDATATYPE sum = pt[0] + pt[pos3] - pt[ pos1 ] - pt[ pos2 ];	

			m_TmpSquaredSum[currentDim] = (float)sum;
		}

		unsigned int count = 0;
		for ( unsigned int currentDim = 0; currentDim < m_NbCovariances; ++currentDim )
		{
			variance[ m_Assignment[count] ] = variance[ m_Assignment[count + 1] ] = 
				area1*( m_TmpSquaredSum[currentDim] - 
				mean[ m_Assignment[count + 2] ]*m_TmpSum[ m_Assignment[count + 3] ] );

			count += 4;
		}

		// *******************************
		// regularization for positive definite matrices, 
		for ( unsigned int currentDim = 0; currentDim < m_Dimension; ++currentDim )
			variance[ currentDim*(m_Dimension+1) ] += 0.001f;

		// *******************************
		// add weight for gaussian signals
		float weight = 2.0f*(m_Dimension+0.1f);
		for ( unsigned int currentDim = 0; currentDim < m_Dimension*m_Dimension; ++currentDim )
			variance[currentDim] *= weight;

		// *******************************
		// use m_TmpSum for cholesky
		valid = ComputeCholeskyDecomposition( variance, m_TmpSum, m_Dimension );

		// *******************************
		// assign diagonal element
		for ( unsigned int currentDim = 0; currentDim < m_Dimension; ++currentDim )
			variance[ currentDim*(m_Dimension+1) ] = m_TmpSum[currentDim];

		for ( unsigned int k = 0; k < m_Dimension; ++k )
			for ( unsigned int i = k+1; i < m_Dimension; ++i )
				variance[ i + k*m_Dimension ] = 0.0f;

		// *******************************
		// sigma point generation, extract columns and +/- the variances
		for ( unsigned int currentDim = 0; currentDim < m_Dimension; ++currentDim )
			sigmaPoints[currentDim] = mean[currentDim];

		for ( unsigned int currentDim = 0; currentDim < m_Dimension*m_Dimension; ++currentDim )
		{
			sigmaPoints[m_Dimension + currentDim ] =   
				 mean[currentDim % m_Dimension ] + variance[currentDim  ];
			sigmaPoints[m_Dimension*m_Dimension + m_Dimension + currentDim ] = 
				 mean[currentDim % m_Dimension ] - variance[ currentDim  ];
		}

		delete[] variance;
		delete[] mean;
	}
    
    	/**
	 * \brief Returns the sigma points in a given roi in one step
	 * sigmapoint vector size: DIM*(2*DIM) ( without mean )
	 */
	void GetReducedSigmaPointFeature( const myRect& roi, float* sigmaPoints, bool& valid )
	{
		// ToDO
		// compute sigma points without mean
	}

	/**
	 * \brief Returns the Number of Covariance Integral Images
	 */
	unsigned int GetNbOfCovIntegralStructures(){ return m_NbCovariances;};

	/**
	 * \brief Returns the Number of Dimensons
	 */
	unsigned int GetDimension(){ return m_Dimension; };

	/**
	 * \brief Returns the Number of Orientation Histogram Bins
	 */
	unsigned int GetNbOfRows(){ return m_NbRows; };

	/**
	 * \brief Returns the Number of Orientation Histogram Bins
	 */
	unsigned int GetNbOfCols(){ return m_NbCols; };

private:

	// general
	unsigned int m_NbElementsPerDimension;
	unsigned int m_NbExtendedElementsPerDimension;
	unsigned int m_Dimension;
	unsigned int m_NbRows;
	unsigned int m_NbCols;

	unsigned int m_NbColsExtended;

	unsigned int m_NbCovariances;

	// defines the compute integral types
	int m_IntegralTypes;

	// compute a cholesky decomposition
	// taken from Numerical Recipes
	bool ComputeCholeskyDecomposition( float *mat, float *p, int dim )
	{
		int i,j,k;                     
		float sum;

		for( i=0; i< dim; ++i )
		{
			for( j=i; j< dim; ++j )
			{
				sum = mat[j+dim*i];
				
				k = i;
				
				while ( --k >= 0 )
					sum -= mat[k+dim*i]*mat[k+dim*j];
		  
				if ( i==j )
				{
					if (sum <= 0.0)
						return false;

					p[i] = sqrt( sum );
				}
				else
					mat[i + dim*j] = sum/p[i];
			}
		}

		return true;
	}

	// compute integral image
	void BuildIntegralSquaredImage( INPUTDATATYPE *inputImage )
	{
		m_IntegralSquaredImage = new INTERNALDATATYPE[m_NbExtendedElementsPerDimension*m_NbCovariances];

		memset( m_IntegralSquaredImage, 0x00, sizeof(INTERNALDATATYPE)*m_NbCovariances*
			m_NbExtendedElementsPerDimension );

		// compute assignment vector
		unsigned int currentPosition = 0;
		for ( unsigned int c1 = 0; c1 < m_Dimension; ++c1 )
		{
			for ( unsigned int c2 = c1; c2 < m_Dimension; ++c2 )
			{
				// store the assignment for efficient variance computation
				m_Assignment[4*currentPosition + 0] = c1*m_Dimension + c2;
				m_Assignment[4*currentPosition + 1] = c2*m_Dimension + c1;
				m_Assignment[4*currentPosition + 2] = c1;
				m_Assignment[4*currentPosition + 3] = c2;

				INPUTDATATYPE *pt1 = &inputImage[ c1*m_NbElementsPerDimension ];
				INPUTDATATYPE *pt2 = &inputImage[ c2*m_NbElementsPerDimension ];
				INTERNALDATATYPE *ptImage = &m_IntegralSquaredImage[ 
					currentPosition*m_NbExtendedElementsPerDimension + m_NbColsExtended + 1];

				for( unsigned int y = 1 ; y < m_NbRows; ++y )
				{	
					for( unsigned int x = 1 ; x < m_NbCols; ++x )
					{
						*ptImage++ = (*pt1++) * (*pt2++);
					}

					pt1++; pt2++; 
					ptImage++; ptImage++;
				}

				currentPosition++;
			}
		}

		// accumulate
		for ( unsigned int currentDim = 0; currentDim < currentPosition; ++currentDim ) 
		{
			INTERNALDATATYPE *pt1 = &m_IntegralSquaredImage[ 
				currentDim*m_NbExtendedElementsPerDimension ];
			INTERNALDATATYPE *pt2 = &m_IntegralSquaredImage[ 
				currentDim*m_NbExtendedElementsPerDimension + 1  ];

			for( unsigned int y = 1 ; y < m_NbRows+1; ++y )
			{	
				for( unsigned int x = 1 ; x < m_NbColsExtended; ++x )
				{
					*pt2 += *pt1;

					pt1++; pt2++;
				}
				pt1++; pt2++; 
			}
		}

		for ( unsigned int currentDim = 0; currentDim < currentPosition; ++currentDim ) 
		{
			INTERNALDATATYPE *pt1 = &m_IntegralSquaredImage[ 
				currentDim*m_NbExtendedElementsPerDimension ];
			INTERNALDATATYPE *pt2 = &m_IntegralSquaredImage[ 
				currentDim*m_NbExtendedElementsPerDimension + m_NbColsExtended  ];

			for( unsigned int y = 1 ; y < m_NbRows+1; ++y )
			{	
				for( unsigned int x = 1 ; x < m_NbColsExtended; ++x )
				{
					*pt2 += *pt1;
					pt1++; pt2++;
				}
				pt1++; pt2++; 
			}
		}
	}

	// computes the squared integral image
	void BuildIntegralImage( INPUTDATATYPE *inputImage )
	{
		m_IntegralImage = new INTERNALDATATYPE[m_Dimension*m_NbExtendedElementsPerDimension];
		memset( m_IntegralImage, 0x00, sizeof(INTERNALDATATYPE)*m_Dimension*
			m_NbExtendedElementsPerDimension );



		for ( unsigned int currentDim = 0; currentDim < m_Dimension; ++currentDim ) 
		{
			INPUTDATATYPE *ptImage = &inputImage[currentDim*m_NbElementsPerDimension];
			INTERNALDATATYPE *ptIntImage = &m_IntegralImage[ currentDim*m_NbExtendedElementsPerDimension 
				+ m_NbColsExtended + 1 ];

			for( unsigned int y = 1 ; y < m_NbRows; ++y )
			{	
				for( unsigned int x = 1 ; x < m_NbCols; ++x )
				{
					*ptIntImage++ = *ptImage++;
				}
				// add for boundary
				ptIntImage++; ptIntImage++; ptImage++;
			}
		}

		// accumulate
		for ( unsigned int currentDim = 0; currentDim < m_Dimension; ++currentDim ) 
		{
			INTERNALDATATYPE *pt1 = &m_IntegralImage[ currentDim*m_NbExtendedElementsPerDimension ];
			INTERNALDATATYPE *pt2 = &m_IntegralImage[ currentDim*m_NbExtendedElementsPerDimension + 1  ];

			for( unsigned int y = 1 ; y < m_NbRows+1; ++y )
			{	
				for( unsigned int x = 1 ; x < m_NbColsExtended; ++x )
				{
					*pt2 += *pt1;

					pt1++; pt2++;
				}
				pt1++; pt2++; 
			}
		}

		for ( unsigned int currentDim = 0; currentDim < m_Dimension; ++currentDim ) 
		{
			INTERNALDATATYPE *pt1 = &m_IntegralImage[ 
				currentDim*m_NbExtendedElementsPerDimension ];
			INTERNALDATATYPE *pt2 = &m_IntegralImage[ 
				currentDim*m_NbExtendedElementsPerDimension + m_NbColsExtended  ];

			for( unsigned int y = 1 ; y < m_NbRows+1; ++y )
			{	
				for( unsigned int x = 1 ; x < m_NbColsExtended; ++x )
				{
					*pt2 += *pt1;
					pt1++; pt2++;
				}
				pt1++; pt2++; 
			}
		}
	}

	// integral image structures
	INTERNALDATATYPE *m_IntegralSquaredImage;
	INTERNALDATATYPE *m_IntegralImage;

	// sum and squared sum tmp structures
	float *m_TmpSum;
	float *m_TmpSquaredSum;

	unsigned int *m_Assignment;
};


