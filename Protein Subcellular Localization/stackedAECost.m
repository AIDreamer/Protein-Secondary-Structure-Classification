function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
m = size(data, 2);
groundTruth = full(sparse(labels, 1:m, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

% Forward propagation: calculate the activation of all layers
a1 = data; % First layer activation is the input itself
a2 = sigmoid(bsxfun(@plus, stack{1}.w * a1, stack{1}.b)); % Inside parentheses is the linear transformation.
a3 = sigmoid(bsxfun(@plus, stack{2}.w * a2, stack{2}.b));
% Calculate a4 Using softmax Regression
z = softmaxTheta * a3;          % 10 x 200 * 200 x 20
z = bsxfun(@minus, z, max(z));
z = exp(z);
sumZ = sum(z);
a4 = bsxfun(@rdivide,z,sumZ);

% Calculate the correctZ by only keeping the transformation of correct
% labels.
correctProb = sum(a4 .* groundTruth);

% Cost function has 2 terms: negative log-likelihood term and weight decay
% of softMaxTheta (there's no need to keep decay other layers)
% Calculate the square error term
log_likelihood = - 1 / m * sum(log(correctProb));
% Calculate the weight decay term (of the softmax theta only)
weightdecay = lambda / 2 * (sum(sum(softmaxTheta.^2)));
% Compute the cost J
cost = log_likelihood + weightdecay;

% Back propagation: Compute the gradient of each theta.
% Compute gradient of the softMaxTheta



% Calculate delta
delta4 = -(groundTruth - a4);
delta3 = softmaxTheta' * delta4 .* dsigmoid(a3);
delta2 = stack{2}.w' * delta3 .* dsigmoid(a2);
% Calculate nabla
nablaW2 = delta3 * a2';
nablab2 = delta3;
nablaW1 = delta2 * a1';
nablab1 = delta2;
% Calculate the desired partial derivatives
softmaxThetaGrad = 1 / m * (a3*delta4')';
% Add the weight decay term for softmaxThetaGrad
softmaxThetaGrad = softmaxThetaGrad + lambda * softmaxTheta;
% Calculate stack grads
stackgrad{2}.w = stackgrad{2}.w + nablaW2 / m;
stackgrad{2}.b = stackgrad{2}.b + sum(nablab2,2) / m;
stackgrad{1}.w = stackgrad{1}.w + nablaW1 / m;
stackgrad{1}.b = stackgrad{1}.b + sum(nablab1,2) / m;

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

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