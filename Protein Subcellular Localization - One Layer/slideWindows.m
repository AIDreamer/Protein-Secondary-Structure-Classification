function [trainingData] = slideWindows(proteinTable,windowSize )
%slideWindows Uses the sliding window technique to obtain substring with
%labels from a set of training protein.
%   The function will take each protein, and take every possible substring
%   of windowSize and use the protein label as the label for the sub
%   string.
%   Return: a table of training data in the form
%           {substring}   [data]
%           {label}       [data]

% Initialize some useful variables
numProteins = size(proteinTable, 2);

% Initialize the training data, the training data
trainingData = cell(2,0);

% Initialize the count of substrings and labels
stringCount = 0;

for proteinNum = 1:numProteins
    % Get the protein data
    data = proteinTable{3,proteinNum};
    label = proteinTable{2,proteinNum};
    
    % Use the protein length to calculate the number of windows
    length = numel(data);
    numWindows = length - windowSize + 1;
    for i = 1:numWindows
        stringCount = stringCount + 1;
        trainingData(:,stringCount) = {data(i:i+windowSize-1),label};
    end
end
