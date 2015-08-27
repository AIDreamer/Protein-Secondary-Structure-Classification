function [sampleData] = convertSamples(sampleWindows, sampleLength)
%convertSamples Transform sample data into binary form
%   Return the sample data into binary form.

% Initialize useful variables
numSamples = size(sampleWindows,2);
aminoAcidBank = ['A';'R';'N';'D';'C';'E';'Q';'G';'H';'I';'L';'K';'M';'F';'P';'S';'T';'W';'Y';'V';'O'];
% 21 proteins, including O as the pseudo-beginner string

% Create an amino acid frame.
aminoAcidFrame = repmat(aminoAcidBank,1,sampleLength);

% Initialize data and label
sampleData = zeros(sampleLength * 21, numSamples);

% Put each sample through each amino acid frame to convert the sub string
% into binary data.
for sampleNum = 1:numSamples
    sampleString = sampleWindows{1,sampleNum};
    
    % Duplicate the string 21 times to get the windows frame
    sampleFrame = repmat(sampleString,21,1);
    
    % Get the binary data
    binaryData = (sampleFrame == aminoAcidFrame);
    sampleData(:,sampleNum) = binaryData(:);
end

end

