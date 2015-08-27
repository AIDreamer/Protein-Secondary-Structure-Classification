function clearedTable = clearOutLittleLabels( proteinTable )
%clearOutLittleLabels delete all proteins with labels that have two few
%samples

% Initialize some useful variable
numProteins = size(proteinTable,2);

% Initialize a clear table
clearedTable = cell(3,0);

% initialize the count of good protein
goodProteinCount = 0;

% Check all proteins and only keep proteins with single label
for proteinNum = 1:numProteins
    data = proteinTable{2,proteinNum};
    if ~(data == 'D') && ~(data == 'V')
        % If protein has exactly one label, then store it
        goodProteinCount = goodProteinCount + 1;
        clearedTable(:,goodProteinCount) = proteinTable(:,proteinNum);
    end
end
end

