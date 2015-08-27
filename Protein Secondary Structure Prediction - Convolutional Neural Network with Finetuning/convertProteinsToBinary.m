function [ proteinTable ] = convertProteinsToBinary( proteinTable )
%convertProteins Convert current protein table into binary form.

% Initialize variables
numProteins = size(proteinTable,2);
aminoAcidBank = ['A';'R';'N';'D';'C';'E';'Q';'G';'H';'I';'L';'K';'M';'F';'P';'S';'T';'W';'Y';'V';'O'];
% 21 proteins, including O as the pseudo-beginner string

for proteinNum = 1:numProteins
    
    proteinData = proteinTable{1,proteinNum};
    proteinLength = numel(proteinData);
    % Create an amino acid frame.
    aminoAcidFrame = repmat(aminoAcidBank,1,proteinLength);

    % create a protein frame
    proteinFrame = repmat(proteinTable{1, proteinNum},21,1);
    
    size(aminoAcidFrame)
    size(proteinFrame)
    % The new data is the binary form
    proteinTable{1,proteinNum} = (aminoAcidFrame == proteinFrame);
end

end

