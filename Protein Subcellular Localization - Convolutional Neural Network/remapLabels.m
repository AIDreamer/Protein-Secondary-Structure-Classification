function [ newLabels ] = remapLabels( labels )
%remapLabels Remap 12 types of protein localization to numeric value from 1
%to 12
%   The labeling process is mapped as followed
%   C --> 1     %   M --> 6
%   E --> 2     %   N --> 7
%   G --> 3     %   P --> 8
%   K --> 4     %   S --> 9
%   L --> 5     %   X --> 10

newLabels = zeros(size(labels)) + 13;
labels = cell2mat(labels);
newLabels(labels == 'C') = 1;
newLabels(labels == 'E') = 2;
newLabels(labels == 'G') = 3;
newLabels(labels == 'K') = 4;
newLabels(labels == 'k') = 4;
newLabels(labels == 'L') = 5;
newLabels(labels == 'M') = 6;
newLabels(labels == 'N') = 7;
newLabels(labels == 'P') = 8;
newLabels(labels == 'S') = 9;
newLabels(labels == 'X') = 10;

end

