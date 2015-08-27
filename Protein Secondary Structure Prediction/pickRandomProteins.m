function [trainingProteins,testingProteins] = pickRandomProteins( proteinTable, numProteins )
%pickRandomProteins Pick a random numof proteins and put it into training
%protein set, put the rest into testing protein set.

% Initialize training proteins sample
trainingProteins = cell(2,0);
testingProteins = proteinTable;

if numProteins < 100
    fprintf('-- Not enough proteins for training');
    return;
end

[trainingProteins,~] = datasample(testingProteins,numProteins,2,'Replace',false);
end

