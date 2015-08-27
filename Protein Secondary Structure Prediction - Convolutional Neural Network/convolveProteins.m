function convolvedProteins = convolveProteins( binaryProteins, W, b )
%convolvedProteins convolve the binary proteins with found features.
%   Use convolution to convolve the binary proteins with found features
%   Input:  a table of binary proteins
%           a set of filters found by the denoising autoencoders.
%   Return: Each protein becomes numFilter x proteinLength matrix, each row
%           represents where each of the filter is in the protein.
%   

numProteins = size(binaryProteins,2);
numFilters = size(W,3);
filterLength = size(W,2);

% Initialize convolved proteins
convolvedProteins = cell(2,numProteins);
convolvedProteins(2,:) = binaryProteins(2,:);

for proteinNum = 1:numProteins
    proteinData = 0+binaryProteins{1,proteinNum};
    proteinLength = size(proteinData,2);
    convolvedFeatures = zeros(numFilters,proteinLength - filterLength + 1);
    
    for filterNum = 1:numFilters
        % Rotate the convolved filters by 180 degrees for the right
        % convolution
        convolvedFilter = rot90(squeeze(W(:,:,filterNum)),2);
        convolvedFeatures(filterNum,:) = conv2(proteinData,convolvedFilter,'valid');
        convolvedFeatures(filterNum,:) = sigmoid(convolvedFeatures(filterNum,:));
    end
    
    convolvedProteins{1,proteinNum} = convolvedFeatures;
end

end
%--------------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which is useful when
% finishing computing convolution.

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

