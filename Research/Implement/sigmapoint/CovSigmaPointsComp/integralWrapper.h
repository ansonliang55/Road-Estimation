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

#define _MYLIB_MEX_H_

#include "mex.h"

#include "IntegralStructures.h"
#include "ObjectHandle.h"

// data types for integral structure
// external data type, represents the input data
typedef float t1;
// internal data type
typedef float t2;

/**
 *  \class MatlabWrapper
 *  \brief This method creates the Integral Structure
 */
void IntegralStructure(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]);

/**
 *  \class MatlabWrapper
 *  \brief This method releases the Integral Structure
 */
void ReleaseIntegralStructure(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]);

/**
 *  \class MatlabWrapper
 *  \brief This method returns typical data from the Integral Structure
 */
void GetFromIntegralStructure(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]);

/**
 *  \class MatlabWrapper
 *  \brief This method returns a full distance map from the Integral Structure
 */
void GetDistanceMapFromIntegralStructure(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]);

/**
 *  \class MatlabWrapper
 *  \brief This method returns a reduced distance map from the Integral Structure
 */
void GetReducedDistanceMapFromIntegralStructure(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]);

/**
 *  \class MatlabWrapper
 *  \brief This method returns distances from the Integral Structure for given ROIs
 */
void GetDistancesFromIntegralStructure(int n_out, mxArray *plhs[], int n_in, const mxArray *prhs[]);



