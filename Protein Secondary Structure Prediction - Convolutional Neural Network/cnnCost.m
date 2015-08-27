function [ cost, grad, preds ] = cnnCost(theta, lambda, windowDim, filterDim,...
                                                trainData, trainLabels, ...
                                                numClasses, numFilters, pred)
                                         
% cnnCost: Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
% Paramters                     
%   theta:       -   Unrolled parameter vector
%   trainData:   -   Training protein substrings in binary format
%   trainLabels: -   A vector containing labels, where labels(i) is the 
%                    label for the i-th training example   
%   numClasses:  -   The number of categories
%   numFilters:  -   Number of convolutional filters
%   pred:        -   Boolean value. True if only wants to forward propagate,
%                    false otherwise
% Returns
%   cost:        -   Cross entropy cost
%   grad:        -   Gradient with respect to theta (if pred==False)
%   preds:       -   List of predictions for each example (if pred==True)

if ~exist('pred','var')
    pred = false;
end;

numImages = size(trainData,3);
convolvedDim = windowDim - filterDim + 1;

%% Reshape parameters and setup gradient matrices

% Wc is 21 x filterLength x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc, Wd, bc] = cnnParamsToStack(theta,windowDim,filterDim,numFilters,numClasses);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_grad = zeros(size(Wc)); % 21 x filterLength x numFilters
Wd_grad = zeros(size(Wd)); % numclasses x (numFilters * convolvedLength)
bc_grad = zeros(size(bc)); % numFilters x 1

% Some other useful variables
m = size(trainData, 3);
groundTruth = full(sparse(trainLabels, 1:m, 1));

%-------------------------------------------------------------
%% Forward propagation: calculate the activation of all layers
% Initialize a2
activations = zeros(convolvedDim, numFilters, numImages);
for filterNum = 1:numFilters
    for imageNum = 1:numImages
        temp = conv2(squeeze(trainData(:,:,imageNum)),rot90(squeeze(Wc(:,:,filterNum)),2),'valid');
        temp = temp + bc(filterNum);
        temp = sigmoid(temp);
        activations(:,filterNum,numImages) = temp(:);
    end
end
activations = reshape(activations,convolvedDim * numFilters,numImages);
% Calculate a3 Using softmax Regression
z = Wd * activations;
z = bsxfun(@minus, z, max(z));
z = exp(z);
sumZ = sum(z);
probs = bsxfun(@rdivide,z,sumZ);
% Calculate the correctZ by only keeping the transformation of correct
% labels.
correctProb = sum(probs .* groundTruth);
% Cost function has 2 terms: negative log-likelihood term and weight decay
% of softMaxTheta (there's no need to keep decay other layers)
% Calculate the square error term
log_likelihood = - 1 / m * sum(log(correctProb));
% Calculate the weight decay term (of the softmax theta only)
weightdecay = lambda / 2 * (sum(sum(Wd.^2)));
% Compute the cost J
cost = log_likelihood + weightdecay;

% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%-------------------------------------------------------
%% Back propagation: Compute the gradient of each theta.
% Compute gradient of the softMaxTheta

a1 = trainData; % First activation is the input data 21 x stringLength
a2 = activations; % Second layer is the convolved activation of the string
a3 = probs;

% Calculate delta
delta3 = -(groundTruth - a3); % 3 x m (softmax error rate)
deltaPool = Wd' * delta3;     % hiddenSize x m (Pooling error rate)
deltaConv = deltaPool .* dsigmoid(a2);
deltaConv = reshape(deltaConv, convolvedDim, numFilters, numImages);

% Calculate nabla.
% Nabla of delta_softmax is not too bad.
nablaWd = delta3 * a2';
% Instantiate zeromatrix of nabla
nablaWc = zeros(21,filterDim,numFilters);
nablabc = zeros(numFilters,1);
% Nabla of convolutional layer involve convolution.
for filterNum = 1:numFilters
    for imageNum = 1:numImages
        nablaWc(:,:,filterNum) = nablaWc(:,:,filterNum) + ... 
            conv2(squeeze(a1(:,:,imageNum)),rot90(squeeze((deltaConv(:,filterNum,imageNum))'),2),'valid');
    end
    temp = deltaConv(:,filterNum,:);
    nablabc(filterNum) = sum(temp(:));
end

%-------------------------------
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

%%% YOUR CODE HERE %%%
Wd_grad = 1 ./ numImages .* nablaWd; % + lambda .* Wd;

for filterNum = 1:numFilters
    Wc_grad(:,:,filterNum) = 1 ./ numImages .* nablaWc(:,:,filterNum); % + lambda .* Wc(:,:,filterNum);
end
bc_grad = 1 ./ numImages .* nablabc;

%% Unroll gradient into grad vector for minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:)];

end

%% SIGMOID FUNCTION -------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

%% DSIGMOID FUNCTION ------------------------------------------------------
% Derivative of sigmoid function sigmoid(z) with respect to a.
% We have f'(z_il) = a_il * (1 - a_il);
% Used to calculate error terms nabla
%
% We use dsigmoid(a) = a * (1 - a) for convenience since a should already
% be calculated from z
function dsigm = dsigmoid(a)
    dsigm = a .* (1 - a);
end