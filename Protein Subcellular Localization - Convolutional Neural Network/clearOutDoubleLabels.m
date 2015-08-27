function clearedTable = clearOutDoubleLabels( proteinTable )
%clearOutDoubleLabels delete all proteins that contain more than 1
%localization type
%   For research purpose, we ignore all proteins that have 2 or more
%   labels.

% Initialize some useful variable
numProteins = size(proteinTable,2);

% Initialize a clear table
clearedTable = cell(3,0);

% initialize the count of good protein
goodProteinCount = 0;

% Check all proteins and only keep proteins with single label
for proteinNum = 1:numProteins
    data = proteinTable{2,proteinNum};
    if numel(data) == 1
        % If protein has exactly one label, then store it
        goodProteinCount = goodProteinCount + 1;
        clearedTable(:,goodProteinCount) = proteinTable(:,proteinNum);
    end
end
end

