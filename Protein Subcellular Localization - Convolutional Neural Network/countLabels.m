function labelCount = countLabels( labels )
%countLabels Count the number of labels of each type and see which type is
%the most rare.

numLabels = size(labels,2);
labelCount = zeros(1,13);

for labelNum = 1:numLabels
    labelCount(labels(labelNum)) = labelCount(labels(labelNum)) + 1;
end

end

