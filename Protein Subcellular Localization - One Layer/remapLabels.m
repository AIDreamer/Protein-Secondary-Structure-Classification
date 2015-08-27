function [ labels ] = remapLabels( labels )
%remapLabels Remap 12 types of protein localization to numeric value from 1
%to 12
%   The labeling process is mapped as followed
%   C --> 1     %   M --> 7
%   D --> 2     %   N --> 8
%   E --> 3     %   P --> 9
%   G --> 4     %   S --> 10
%   K --> 5     %   V --> 11
%   L --> 6     %   X --> 12

labels = cell2mat(labels);
labels(labels == 'C') = 1;
labels(labels == 'D') = 2;
labels(labels == 'E') = 3;
labels(labels == 'G') = 4;
labels(labels == 'K') = 5;
labels(labels == 'L') = 6;
labels(labels == 'M') = 7;
labels(labels == 'N') = 8;
labels(labels == 'P') = 9;
labels(labels == 'S') = 10;
labels(labels == 'V') = 11;
labels(labels == 'X') = 12;

end

