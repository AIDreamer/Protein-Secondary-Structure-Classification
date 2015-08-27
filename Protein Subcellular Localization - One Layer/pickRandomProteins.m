function [trainingProteins,testingProteins] = pickRandomProteins( proteinTable, numProteins )
%pickRandomProteins Pick a random numof proteins and put it into training
%protein set, put the rest into testing protein set.
%   choose random proteins for training set, make sure the protein contains
%   all 12 available labels.

% Initialize training proteins sample
trainingProteins = cell(3,0);
testingProteins = cell(3,0);

if numProteins < 100
    fprintf('-- Not enough proteins for training');
    return;
end

while checkNumLabels(trainingProteins) == 0
    testingProteins = proteinTable;
    % Pick a random number of proteins
    [trainingProteins,~] = datasample(testingProteins,numProteins,2,'Replace',false);
end

end

