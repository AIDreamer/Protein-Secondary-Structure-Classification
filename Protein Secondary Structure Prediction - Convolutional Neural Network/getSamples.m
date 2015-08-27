function sampleData = getSamples(proteinTable,sampleSize)
%slideWindows Uses the sliding window technique to obtain substring with
%labels from a set of training protein.
%   The function will take each protein, and take every possible substring
%   of windowSize and use the protein label as the label for the sub
%   string.
%   Input:
%   * A lengthened protein table
%   * windowSize
%   * sampleSize
%   Return: a table of training data in the form
%           {substring}   [data]
%           {label}       [data]

% Initialize some useful variables
numProteins = size(proteinTable, 2);

% Initialize the training data, the training data
sampleData = cell(1,0);

% Initialize the count of substrings and labels
sampleCount = 0;

for proteinNum = 1:numProteins
    % Get the protein data
    data = proteinTable{1,proteinNum};
    % Calculate the number of sample for each lengthened Protein
    numSamples = numel(data) - sampleSize + 1;

    for i = 1:numSamples
        sampleCount = sampleCount + 1;
        sampleData{sampleCount} = data(i:i+sampleSize-1);
    end
        
end
