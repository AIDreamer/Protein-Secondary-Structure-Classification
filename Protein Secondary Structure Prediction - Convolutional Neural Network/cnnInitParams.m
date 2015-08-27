function theta = cnnInitParams(imageDim,filterDim,numFilters,numClasses)
% Initialize parameters for a single layer convolutional neural
% network followed by a softmax layer.
%                            
% Parameters:
%  imageDim   -  height/width of image
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  numClasses -  number of classes to predict
%
%
% Returns:
%  theta      -  unrolled parameter vector with initialized weights

%% Initialize parameters randomly based on layer sizes.
assert(filterDim < imageDim,'filterDim must be less that imageDim');

Wc = 1e-1*randn(21,filterDim,numFilters);

outDim = imageDim - filterDim + 1; % dimension of convolved image

hiddenSize = outDim*numFilters;

% we'll choose weights uniformly from the interval [-r, r]
r  = sqrt(6) / sqrt(numClasses+hiddenSize+1);
Wd = rand(numClasses, hiddenSize) * 2 * r - r;

bc = zeros(numFilters, 1);

% Convert weights and bias gradients to the vector form.
% This step will "unroll" (flatten and concatenate together) all 
% your parameters into a vector, which can then be used with minFunc. 
theta = [Wc(:) ; Wd(:) ; bc(:)];

end

