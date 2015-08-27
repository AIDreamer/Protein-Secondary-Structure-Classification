function proteinType = predictType(protein, Wc, bc, softmaxModel, convolvedDim)
%predictType predict the localization type of the given protein.

% Initialize some useful variables
numFilters = size(Wc,3);

% Convolved the proteins with known filters and slide them into smaller
% piece.
protein = convertProteinsToBinary(protein);
convolvedProtein = convolveProteins(protein, Wc, bc);
data = slideConvolvedProteins(convolvedProtein, convolvedDim, numFilters);

% use softmax model to predict labels.
pred = softmaxPredict(softmaxModel, data);
% The prediction is the most popular label in the list.
proteinType = mode(pred);

end

