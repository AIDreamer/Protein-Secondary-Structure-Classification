function proteinTable = injectNoise(proteinTable,percentage)
%injectNoise Add more noise into the protein data based on available
%percentage.
%  Noise injection usually prove to be helpful for improving performance of
%  the prediction model

% Initialize some useful variables
numProteins = size(proteinTable,2);

% for each protein in the table, inject some random noise
for proteinNum = 1:numProteins
    proteinData = proteinTable{3,proteinNum};
    % Calculate the number of noise addition.
    dataLength = length(proteinData);
    noise = round(dataLength * percentage);
    proteinData(randperm(dataLength,noise)) = 'O';
    proteinTable{3,proteinNum} = proteinData;
end

