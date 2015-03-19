clear all;
close all;

% test package CovSigmaPointsComp
% using only 3 channels of a color image

integralWrapper

im = imread('GrazUhrTurm.jpg');

% convert to double!
% add some noise in order to handle homogenous areas
im = double( im ) + 2.0*randn( size(im) );

% build integral structure
obj = integralWrapper('buildIntegralStructure',  im );

% specify a ROI in the image
roiReference = [ 240 125 65 30 ];
%roiReference = [ 253 40 21 17 ];

% first test
% sum all values within the given ROI using the wrapper
sumValues = integralWrapper('getFromIntegralStructure', 'sumIntImage', obj, roiReference );

% sum all values within the given ROI using matlab
s1 = sum(sum( im( roiReference(1)+1 : roiReference(1) + roiReference(3), ...
           roiReference(2)+1 : roiReference(2) + roiReference(4), 1 ) ) );
s2 = sum(sum( im( roiReference(1)+1 : roiReference(1) + roiReference(3), ...
           roiReference(2)+1 : roiReference(2) + roiReference(4), 2 ) ) );
s3 = sum(sum( im( roiReference(1)+1 : roiReference(1) + roiReference(3), ...
           roiReference(2)+1 : roiReference(2) + roiReference(4), 3 ) ) );

% output
disp( sprintf('Matlab Sum Evaluation: %f %f %f', s1,s2,s3) ) ;
disp( sprintf('Wrapper Sum Evaluation: %f %f %f', sumValues) );


% second test
% mean values within the given ROI using the wrapper
meanValues = integralWrapper('getFromIntegralStructure', 'meanIntImage', obj, roiReference );

% mean values within the given ROI using matlab
m1 = mean2( im( roiReference(1)+1 : roiReference(1) + roiReference(3), ...
           roiReference(2)+1 : roiReference(2) + roiReference(4), 1 ) );
m2 = mean2( im( roiReference(1)+1 : roiReference(1) + roiReference(3), ...
           roiReference(2)+1 : roiReference(2) + roiReference(4), 2 ) );
m3 = mean2( im( roiReference(1)+1 : roiReference(1) + roiReference(3), ...
           roiReference(2)+1 : roiReference(2) + roiReference(4), 3 ) );

% output
disp( sprintf('Matlab Mean Evaluation: %f %f %f', m1,m2,m3) ) ;
disp( sprintf('Wrapper Mean Evaluation: %f %f %f', meanValues) );

% third test
% extract mean and covariance features
meanAndCov = integralWrapper('getFromIntegralStructure', 'meanAndVariance', obj, roiReference );
tmpData =  im( roiReference(1)+1 : roiReference(1) + roiReference(3), ...
               roiReference(2)+1 : roiReference(2) + roiReference(4), : );
  
meanVectorMatlab = mean( reshape( tmpData, [ size(tmpData,1)*size(tmpData,2) 3 ] ) )
meanVectorWrapper = meanAndCov( 1:3 )

covarianceMatrixMatlab = cov( reshape( tmpData, [ size(tmpData,1)*size(tmpData,2) 3 ] ) )
covarianceMatrixWrapper = reshape( meanAndCov( 4:end ), [3 3] )

% fourth test, find minima in the response map using Sigma Points
% extract reference sigma point vector
resRef = integralWrapper('getFromIntegralStructure', 'covarianceSigma', ...
    obj , roiReference );

% compute Euclidean distance map using the reference sigma point vector
% see C++ code for details
responseMap = integralWrapper('getDistanceMapFromIntegralStructure', ...
    'covarianceSigma', obj , [roiReference(3) roiReference(4)], resRef );

% here we use an image cropping due to value assignment at (left,upper) 
% rectangle position

%figure, imshow( uint8(im(1:end-roiReference(3), 1:end-roiReference(4),:)), ...
%    [] ), hold on, title('RGB Image');
%rectangle( 'Position', [ roiReference(2), roiReference(1), roiReference(4), ...
%    roiReference(3) ],  'LineWidth', 3, 'EdgeColor','r' );

% toDo: normalize response map with e.g. exp, etc
% figure, imshow( responseMap(1:end-roiReference(3), 1:end-roiReference(4)), [] ), title('Response Map');

% find min distance position
[dummy, xPos ] = min( min( responseMap, [], 1 ) );
[dummy, yPos ] = min( min( responseMap, [], 2 ) );

% -1 for Matlab,C++ index conversion!
disp( sprintf('Minima at (%d, %d)', yPos-1, xPos-1 ) )
disp( sprintf('ROI at (%d, %d)', roiReference(1:2) ) )

% release integral images
integralWrapper('releaseIntegralStructure', obj );
