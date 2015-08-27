%% Using Stacked Autoencoders to classify Protein Subcellular Localization
%  This method extracts protein substring using sliding window method and
%  then use Stacked Autoencoders to classify

%%======================================================================
%% Step 0: Initialization
%  Here we initialize some parameters used for the exercise.

%  Clear everything off the screne
clear ; close all; clc;
fprintf('PART 0: INITIALIZATION\n');
fprintf('======================\n');
fprintf('-- Initializing\n');

%%======================================================================
%% Step 1: Read the protein file
%  Here we initialize some parameters used for the exercise.

fprintf('\nPART 1: READ THE PROTEIN FILE\n');
fprintf('=============================\n');
%  Put the whole protein file into a text scanner and scan all the file in
%  to a cell array.
fprintf('-- Scanning the whole protein data and store it in a cell...\n');
fileID = fopen('proteinFile.fa');
proteinCell = textscan(fileID,'%s');
fclose(fileID);
proteinCell = proteinCell{1};

%  Process the proteinCells and put it in the protein table with the format
%  {Protein Name} {Localization Type} {Protein String}. Keep the double
%  type.
fprintf('-- Producing a 3 x m protein table in the form:\n');
fprintf('   {Protein name}        [data]\n');
fprintf('   {Localization type}   [data]\n');
fprintf('   {Protein string}      [data]\n');
proteinTable = processProteins(proteinCell);
fprintf('-- For research purpose, clearing out all double label protein and proteins with D and V lables...\n');
proteinTable = clearOutDoubleLabels(proteinTable);
proteinTable = clearOutLittleLabels(proteinTable);

%% Step 2: Process the protein
%  Here we split the proteins we have into training and testing sets as
%  well as slide the windows to get the substring data.

fprintf('\nPART 2: READ THE PROTEIN FILE\n');
fprintf('=============================\n');

% Splitting proteins into training set and testing set
numProteins = size(proteinTable,2);
trainSize = 100;
fprintf('-- Randomly pick proteins into training set and testing set...\n');
[trainProteins, testProteins] = pickRandomProteins(proteinTable,trainSize);
fprintf('   # of training proteins: %d\n', size(trainProteins,2) );
fprintf('   # of testing proteins: %d\n', numProteins - size(trainProteins,2));

% Ensuring that the training set have datas of all labels
if checkNumLabels(trainProteins)
    fprintf('   Training datas contain all 12 labels\n');
else
    fprintf('   Training datas DOES NOT contain enough labels\n');
end

% Inject noise to training proteins
noisePercentage = 0.25;
noisyProteins = injectNoise(trainProteins,noisePercentage);

% Slide protein into substring of some arbitrary size
windowDim = 15;
filterDim = 6;
convolvedDim = windowDim - filterDim + 1;
windowSize = filterDim * 20;
filterSize = filterDim * 20;
fprintf('-- Sampling small string of filter size %d...\n', filterDim);
noisyWindows = slideWindows(noisyProteins,filterDim);
trainWindows = slideWindows(trainProteins,filterDim);

% Convert the new training data into binary data with label attached
fprintf('-- Converting each sample data into binary data...\n');
fprintf('   Each sample will be represent by %d x 20 = %d binary values\n',filterDim, filterDim * 20);
[noisyData,trainLabels] = convertWindows(noisyWindows,filterDim);
[trainData,~] = convertWindows(trainWindows,filterDim);
testLabels = testProteins(2,:);

% Remap the letter label to numeric label (to make it easier for the
% softmax classifer to do its job
trainLabels = remapLabels(trainLabels);
testLabels = remapLabels(testLabels);

%---------------------------------------------
%% Step 3: Detect first-level protein features
%  Detect new features out of the set of training data using the sparse
%  autoencoder
fprintf('\nPART 3: DETECT PROTEIN FEATURES\n');
fprintf('=============================\n');
addpath 'Sparse Autoencoder'

%  Set up some useful variables
fprintf('-- Randomly initializing theta of the first hidden layer...\n');

numFilters = 60;       % number of hidden units of layer 1
sparsityParam = 0.2;      % desired average activation of the hidden units.
lambda = 0.0001;          % weight decay parameter       
beta = 3;                 % weight of sparsity penalty term

%  Randomly initialize the parameters
sae1Theta = initializeParameters(numFilters, filterSize);

% Set sae1OptTheta equals to the initialialized sae1Theta, this will become
% our trained parameters.
sae1OptTheta = sae1Theta;

%  Use minFunc to minimize the function
fprintf('-- Detecting features...\n');
addpath 'Sparse Autoencoder/minFunc/'
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 1000;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';



tic;
[sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   filterSize, numFilters, ...
                                   lambda, sparsityParam, ...
                                   beta, noisyData, trainData), ...
                              sae1Theta, options);
toc;

% Visualizing found features.
Wc = reshape(sae1OptTheta(1:numFilters*filterSize), filterSize, numFilters);
bc = sae1OptTheta(2*numFilters*filterSize + 1 : 2*numFilters*filterSize + numFilters);

figure(1);
displayProteinFeatures(Wc, 1, 1, 20, 0, filterDim, 20);
title('First-level features')
set(gcf,'name','window title string','numbertitle','off')

%----------------------------------------------------------------
%% Step 4: Convolve the found features with current training data
fprintf('\nPART 4: CONVOLVE FOUND FEATURES WITH TRAINING DATA\n');
fprintf('==================================================\n');

fprintf('-- Converting proteins from letter form to binary form...\n');
% convert filters to 21 x sampleLength image form
Wc = reshape(Wc,20,filterDim,numFilters);

trainProteins2 = trainProteins;
trainProteins = convertProteinsToBinary(trainProteins);

fprintf('-- Convolved the protein with found filters...\n');
convolvedTrainProteins = convolveProteins(trainProteins, Wc, bc);

fprintf('-- Slide proteins into smaller substring. These will be "convolved strings"...\n');
[convolvedData,convolvedLabels] = slideConvolvedProteins(convolvedTrainProteins, convolvedDim, numFilters);
convolvedLabels = remapLabels(convolvedLabels);

%-------------------------------------
%% Step 5: Train the softmax classifer
%  This trains the softmax classifier for determining the final labels
fprintf('\nPART 5: TRAIN THE SOFTMAX CLASSIFIER\n');
fprintf('====================================\n');
addpath 'Softmax Classifier'

hiddenSize = convolvedDim * numFilters;
                                    
fprintf('-- Randomly initialize theta of the softmax classifier...\n');
% Randomly initialize the parameters
numClasses = 10;
softmaxTheta = 0.005 * randn(hiddenSize * numClasses, 1);

% Train the softmax classifier, the classifier takes in input of dimension 
% "hiddenSizeL2" corresponding to the hidden layer size of the 2nd layer.
% You should store the optimal parameters in saeSoftmaxOptTheta

fprintf('-- Train the softmax classifier based on the data and train labels given...\n');
softmaxModel = softmaxTrain(hiddenSize, numClasses, lambda, ...
                            convolvedData, convolvedLabels, options);
                        
softmaxOptTheta = softmaxModel.optTheta(:);

%--------------
%% Step 7: Test
% Using the new trained system to predict protein subcellular localization
% in the testing set.
fprintf('\nPART 6: TEST THE WHOLE NETWORK\n');
fprintf('==============================\n');
fprintf('-- Testing on the training set...\n');

numTrainProteins = size(trainProteins2,2);
labels = trainProteins2(2,:);
labels = remapLabels(labels);
pred = zeros(1,numTestProteins);
for proteinNum = 1:numTrainProteins
    protein = trainProteins2(:,proteinNum);
    pred(proteinNum) = predictType(protein, Wc, bc, softmaxModel, convolvedDim);
end

acc = mean(pred == labels);

fprintf('   Accuracy on training test: %0.3f%%\n', acc * 100);

% Do the same process for the testing set
fprintf('-- Testing on the testing set...\n');

numTestProteins = size(testProteins,2);
labels = testProteins(2,:);
labels = remapLabels(labels);
pred = zeros(1,numTestProteins);
for proteinNum = 1:numTestProteins
    protein = testProteins(:,proteinNum);
    pred(proteinNum) = predictType(protein, Wc, bc, softmaxModel, convolvedDim);
end

acc = mean(pred == labels);

fprintf('   Accuracy on testing set: %0.3f%%\n', acc * 100);