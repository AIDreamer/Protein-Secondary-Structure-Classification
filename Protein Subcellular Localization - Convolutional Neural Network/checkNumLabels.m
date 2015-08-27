function labelCheck = checkNumLabels( proteinTable )
%checkNumLabels Ensure that the database chosen contains all 10 labels.
%   Return true if the number of labels equal 10, false otherwise

% Initialize useful variables
numProteins = size(proteinTable,2);
if numProteins < 10
    labelCheck = false;
    return;
end

% Create an alphabet of all labels
labelTypes = 'CEGKLMNPSX';
labelTypes = labelTypes(:);
labelTypes = repmat(labelTypes,1,numProteins);

% Create a table of all labels
allLabels = proteinTable(2,:);
allLabels = repmat(allLabels,10,1);
allLabels = cell2mat(allLabels);

% Check if each label type exist
checkLabels = allLabels == labelTypes;
checkLabels = sum(checkLabels,2);
checkLabels = checkLabels > 0;
labelCheck = (sum(checkLabels) == 10);

end

