function proteinTable = processProteins( proteinCell )
%ProcessProteins Process the raw Cell data into a protein table with a
%better format
%   Input: {m x 1} cell array of m lines of protein data
%   Return: a proteinTable with a format as followed
%   {protein name} {localization type} {protein string}

% STEP 1: INITIALIZATION
% Initialize the number of lines
numLines = size(proteinCell,1);

% Initialize the proteinTable
proteinTable = cell(3,0);

% initialize protein count (to later squeeze the data)
proteinCount = 0;

% Initialize some data field for data collection (not necessary)
proteinName = '';
proteinType = '';
proteinString = '';

% Initialize protein status, there are 3 status available
% status = 1: just read the protein name
% status = 2: just read the protein type
% status = 3: reading the protein string
% status initialize = 3
status = 3;

% STEP 2: Read the 1-D Cell into proper protein table

for lineNum = 1:numLines
    lineData = proteinCell{lineNum};
    if lineData(1) == '>'
        % If found a new protein, store all current data in a string and
        % reset all data field
        proteinCount = proteinCount + 1;
        proteinTable(:,proteinCount) = {proteinName,proteinType,proteinString};
        proteinName = '';
        proteinType = '';
        proteinString = '';
        % Read the protein name
        proteinName = lineData(2:end);
        status = 1;
    elseif status == 1;
        proteinType = lineData;
        status = 2;
    elseif status == 2;
        proteinString = lineData;
        status = 3;
    elseif status == 3;
        proteinString = strcat(proteinString,lineData);
    end
    
% Store the last protein
proteinTable(:,proteinCount) = {proteinName,proteinType,proteinString};

end

