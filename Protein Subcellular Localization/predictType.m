function proteinType = predictType(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, protein, windowSize)
%predictType predict the localization type of the given protein.

% Create small substring from the protein and make it become the new
% dataset.
proteinWindows = slideWindows(protein,windowSize);
[proteinData,~] = convertWindows(proteinWindows,windowSize);

[allProb] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, proteinData);

% The proteinType would be the most popular type in the string
[~,pred] = max(sum(allProb,2));
proteinType = pred;

end

