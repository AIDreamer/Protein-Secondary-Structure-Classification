function [ proteinTable ] = lengthenProteins( proteinTable, windowLength )
%lengthenProtein Make the protein longer according to the size of each
%window
%   Return the new protein table taht contains some padded O at the
%   beginning and the end of the proteins

% Initialize useful variables
numProteins = size(proteinTable,2);
% Calculate the number of padded O
addedAcids = repmat('O', 1, floor(windowLength / 2));

% Pad the added O into each proteins
for proteinNum = 1:numProteins
    proteinString = proteinTable{1,proteinNum};
    proteinTable{1,proteinNum} = strcat(addedAcids, proteinString, addedAcids);
end

end

