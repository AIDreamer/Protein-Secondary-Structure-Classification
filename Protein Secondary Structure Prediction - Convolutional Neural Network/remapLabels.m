function [ newLabels ] = remapLabels( labels )
%remapLabels Remap 12 types of protein localization to numeric value from 1
%to 12
%   The labeling process is mapped as followed
%   C --> 1  Coil
%   E --> 2  Beta Strand
%   H --> 3  Alpha Helix

newLabels = zeros(size(labels));
newLabels(labels == 'C') = 1;
newLabels(labels == 'E') = 2;
newLabels(labels == 'H') = 3;

end

