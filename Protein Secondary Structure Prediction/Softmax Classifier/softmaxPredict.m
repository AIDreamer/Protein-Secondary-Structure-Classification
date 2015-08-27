function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix

% Zero-instantiate the prediction
pred = zeros(1, size(data, 2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

% Calculate all linear transformation z. The resulting matrix will be of size
% 10 x m. Each 10-dimensional column represent the linear transformation of 
% an input x with each class from 1 to 10.
z = theta * data; % 10 x 60000

% Subtract each transformation by the highest transformation of the input
% (to prevent overflow)
z = bsxfun(@minus, z, max(z));

% exponentiate z
z = exp(z);

% Compute the sum of each exponential
sumZ = sum(z);

% Calculate all probability
allProb = bsxfun(@rdivide,z,sumZ);

% Take the label with highest probability as prediction
[~,pred] = max(allProb);





% ---------------------------------------------------------------------

end

