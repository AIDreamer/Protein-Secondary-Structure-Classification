function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

% Initialize some useful variables
m = size(data,2); % size of training set
rho = sparsityParam; % for convenience and conventional notation

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

% Forward propagation: calculate the activation of all layers
a1 = data; % First layer activation is the input itself
a2 = sigmoid(bsxfun(@plus, W1 * a1, b1)); % Inside parentheses is the linear transformation.
a3 = sigmoid(bsxfun(@plus, W2 * a2, b2)); % Note: h(data) = a3
% Cost function has 3 terms: square error term, weight decay term and
% sparsity constraint term.
% Calculate the square error term
squarederror = 1 / (2 * m) * sum(sum((a3 - data) .^ 2));
% Calculate the weight decay term
weightdecay = lambda / 2 * (sum(sum(W1.^2)) + sum(sum(W2.^2)));
% Calculate the sparsity constraint term.
% p-hat: average activation of each hidden unit (over the whole training set)
% we want most of them to be zero so that hidden units are "sparse."
rho_hat = 1 / m  * sum(a2,2);
% Kullback-Leibler divergence
KLdivergence = sum(rho .* log(rho./rho_hat) + (1-rho) .* log((1-rho)./(1-rho_hat)));
sparsity = beta * KLdivergence;
% Cost function is the sum of all three terms.
cost = squarederror + weightdecay + sparsity;

% Back propagation: Calculate error term nabla
y = data; % We wish to reconstruct the input as perfectly as possible.
delta3 = -(y - a3) .* dsigmoid(a3);
delta2 = bsxfun(@plus, W2' * delta3, beta .* (-rho./rho_hat + (1-rho)./(1-rho_hat)));
delta2 = delta2 .* dsigmoid(a2);

% Calculate nabla, which is the desired partial derivatives
nablaW2 = delta3 * a2';
nablab2 = delta3;
nablaW1 = delta2 * a1';
nablab1 = delta2;

% Calculate the desired partial derivatives
W2grad = W2grad + nablaW2 / m + lambda * W2;
b2grad = b2grad + sum(nablab2,2) / m;
W1grad = W1grad + nablaW1 / m + lambda * W1;
b1grad = b1grad + sum(nablab1,2) / m;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

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