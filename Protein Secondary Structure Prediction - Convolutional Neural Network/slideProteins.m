function [data, labels] = slideProteins( proteins, windowLength )
%slideProteins slide the lengthedn ProteinBank into small substrings
%   This functions will slide the initial 21 x m proteins into smaller
%   substrings of windowLength

% Initialize useful variables
numProteins = size(proteins,2);

% Calculate the total number of string
numStrings = 0;
for proteinNum = 1:numProteins
    numStrings = numStrings + size(proteins{2,proteinNum},2);
end

% Initialize two data field
data = zeros(21,windowLength,numStrings);
labels = zeros(1,numStrings);

stringCount = 0;
% Cut data into small window string
for proteinNum = 1:numProteins
    proteinData = proteins{1,proteinNum};
    labelData = proteins{2,proteinNum};
    numLabels = numel(labelData);
    for labelNum = 1:numLabels
        stringCount = stringCount + 1;
        temp = proteinData(:,labelNum:labelNum+windowLength-1);
        data(:,:,stringCount) = temp;
        labels(stringCount) = labelData(labelNum);
    end
end

end

