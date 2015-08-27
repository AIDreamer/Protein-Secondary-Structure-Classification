function [data,labels] = slideConvolvedProteins( convolvedProteins,windowLength, sampleLength, hiddenSizeL1)
%slideProteins Slide each protein into smaller string and attach label into
%it.
%   After convolving proteins, we can now slide proteins into even smaller
%   strings and use them for pooling and classification.

% Initialize useful variables
numProteins = size(convolvedProteins,2);
stringSize = windowLength - sampleLength + 1;

% Calculate the total number of string
numStrings = 0;
for proteinNum = 1:numProteins
    numStrings = numStrings + size(convolvedProteins{2,proteinNum},2);
end
% Initialize two data field
data = zeros(hiddenSizeL1*stringSize,numStrings);
labels = zeros(1,numStrings);

stringCount = 0;
% Cut data into small window string
for proteinNum = 1:numProteins
    proteinData = convolvedProteins{1,proteinNum};
    labelData = convolvedProteins{2,proteinNum};
    numLabels = numel(labelData);
    for labelNum = 1:numLabels
        stringCount = stringCount + 1;
        temp = proteinData(:,labelNum:labelNum+stringSize-1);
        data(:,stringCount) = temp(:);
        labels(stringCount) = labelData(labelNum);
    end
end

