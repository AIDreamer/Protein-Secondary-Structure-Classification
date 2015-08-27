function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
% theta - 10 x N weight matrix containing weights connecting to input data
% for each of 10 classes.

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

% initialize the number of cases.
m = size(data, 2);

% Initialize the ground truth
groundTruth = full(sparse(labels, 1:m, 1));

% Zero-instantiate cost and thetagrad
cost = 0;
thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

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

% Calculate the correctZ by only keeping the transformation of correct
% labels.
correctProb = sum(allProb .* groundTruth);

% Compute the cost J
cost = - 1 / m * sum(log(correctProb));
% Add weight decay term
cost = cost + lambda / 2 * sum(sum(theta.^2));

% Compute gradient
thetagrad = - 1 / m * (data*(groundTruth - allProb)')' + lambda * theta;
    
% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

