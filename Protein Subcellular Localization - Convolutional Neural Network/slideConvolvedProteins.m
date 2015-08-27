function [data,labels] = slideConvolvedProteins( convolvedProteins, convolvedDim, numFilters)
%slideProteins Slide each protein into smaller string and attach label into
%it.
%   After convolving proteins, we can now slide proteins into even smaller
%   strings and use them for pooling and classification.

% Initialize useful variables
numProteins = size(convolvedProteins,2);

% Calculate the total number of string
numStrings = 0;
for proteinNum = 1:numProteins
    numStrings = numStrings + size(convolvedProteins{1,proteinNum},2) - convolvedDim + 1;
end
% Initialize two data field
data = zeros(numFilters*convolvedDim,numStrings);
labels = cell(1,numStrings);

stringCount = 0;
% Cut data into small window string
for proteinNum = 1:numProteins
    proteinData = convolvedProteins{1,proteinNum};
    labelData = convolvedProteins{2,proteinNum};
    numLabels = size(proteinData,2) - convolvedDim + 1;
    for labelNum = 1:numLabels
        stringCount = stringCount + 1;
        temp = proteinData(:,labelNum:labelNum+convolvedDim-1);
        data(:,stringCount) = temp(:);
        labels{stringCount} = labelData;
    end
end

