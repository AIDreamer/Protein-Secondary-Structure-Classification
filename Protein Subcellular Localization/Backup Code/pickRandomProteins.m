function [trainProteins,testProteins] = pickRandomProteins( proteinTable, numProteinsPerLabel )
%pickRandomProteins Pick a random numof proteins and put it into training
%protein set, put the rest into testing protein set.
%   choose random proteins for training set, make sure the protein contains
%   all 12 available labels.

% Initialize training proteins sample
numProteins = size(proteinTable,2);
trainProteins = cell(3,12*numProteinsPerLabel);
testProteins = cell(3,numProteins - size(trainProteins,2));

% Separate proteins by labels
% Create an alphabet of all labels
labelTypes = 'CDEGKLMNPSVX';
labelTypes = labelTypes(:);
labelTypes = repmat(labelTypes,1,numProteins);

% Create a table of all labels
allLabels = cell2mat(proteinTable(2,:));
allLabels = repmat(allLabels,12,1);

% Available proteins
availableProteins = labelTypes == allLabels;

% Create a cell contain all possible proteins separated by label type
allProteins = cell(1,12);
numProteinsType = zeros(1,12); % This array contains the number of proteins in each type.
for proteinType = 1:12
    allProteins{proteinType} = proteinTable(:,availableProteins(proteinType,:) == 1);
    numProteinsType(proteinType) = sum(availableProteins(proteinType,:),2);
end

% Pick a certain number of proteins with each type
for proteinType = 1:12
    % Calculate begin point and end point in test proteins data base
    endPoint = sum(numProteinsType(1:proteinType));
    beginPoint = endPoint - numProteinsType(proteinType) + 1;
    % All locate new train proteins and new test proteins
    [newTrainProteins,~] = datasample(allProteins{proteinType},numProteinsPerLabel,2,'Replace',false);
    trainProteins(:,numProteinsPerLabel*(proteinType-1) + 1 : numProteinsPerLabel*proteinType) = newTrainProteins;
    testProteins(:,beginPoint:endPoint) = allProteins{proteinType};
end

while checkNumLabels(trainProteins) == 0
    testProteins = proteinTable;
    % Pick a random number of proteins
    [trainProteins,~] = datasample(testProteins,numProteinsPerLabel,2,'Replace',false);
end

end

