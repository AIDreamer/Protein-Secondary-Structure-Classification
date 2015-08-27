function [data,label] = convertWindows(windows, windowLength)
%convertWindows convert the substring data into binary value.
%   Since each character or each amino acid can be treated as a 
%   20-dimensional binary vector. Convert all character into 20 binary 
%   value and make them the data.

% Initialize useful variables
numWindows = size(windows,2);
aminoAcidBank = ['A';'R';'N';'D';'C';'E';'Q';'G';'H';'I';'L';'K';'M';'F';'P';'S';'T';'W';'Y';'V'];

% Create an amino acid frame.
aminoAcidFrame = repmat(aminoAcidBank,1,windowLength);

% Initialize data and label
label = windows(2,1:end);
data = zeros(windowLength * 20, numWindows);

% Put each window through each amino acid frame to convert the sub string
% into binary data.
for windowNum = 1:numWindows
    windowString = windows{1,windowNum};
    
    % Duplicate the string 20 times to get the windows frame
    windowFrame = repmat(windowString,20,1);
    
    % Get the binary data
    binaryData = (windowFrame == aminoAcidFrame);
    data(:,windowNum) = binaryData(:);
end

end

