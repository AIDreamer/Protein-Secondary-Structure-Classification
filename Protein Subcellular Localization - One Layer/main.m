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
fprintf('-- For research purpose, clearing out all double label protein...\n');
proteinTable = clearOutDoubleLabels(proteinTable);

%% Step 2: Process the protein
%  Here we split the proteins we have into training and testing sets as
%  well as slide the windows to get the substring data.

fprintf('\nPART 2: READ THE PROTEIN FILE\n');
fprintf('=============================\n');

% Splitting proteins into training set and testing set
numProteins = size(proteinTable,2);
trainSize = 250;
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
trainProteins = injectNoise(trainProteins,noisePercentage);

% Slide protein into substring of some arbitrary size
windowSize = 8;
dataSize = windowSize * 20;
fprintf('-- Sliding windows into substring of size %d...\n', windowSize);
trainingWindows = slideWindows(trainProteins,windowSize);

% Convert the new training data into binary data with label attached
fprintf('-- Converting string data into binary data...\n');
fprintf('   Each substring will be represent by %d x 20 = %d binary values\n',windowSize, windowSize * 20);
[trainData,trainLabels] = convertWindows(trainingWindows,windowSize);
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

inputSize = dataSize;     % number of input units 
hiddenSizeL1 = 140;       % number of hidden units of layer 1 
sparsityParam = 0.2;      % desired average activation of the hidden units.
lambda = 0.0001;          % weight decay parameter       
beta = 3;                 % weight of sparsity penalty term

%  Randomly initialize the parameters
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);

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
                                   inputSize, hiddenSizeL1, ...
                                   lambda, sparsityParam, ...
                                   beta, trainData), ...
                              sae1Theta, options);
toc;

% Visualizing found features.
features1 = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), hiddenSizeL1, inputSize);

figure(1);
displayProteinFeatures(features1', 1, 1, 20, 0, windowSize, 20);
title('First-level features')
set(gcf,'name','window title string','numbertitle','off')

%-------------------------------------
%% Step 5: Train the softmax classifer
%  This trains the softmax classifier for determining the final labels
fprintf('\nPART 5: TRAIN THE SOFTMAX CLASSIFIER\n');
fprintf('====================================\n');
addpath 'Softmax Classifier'

% This part re-represent the original input as features found in
% sae2OptTheta
[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainData);
                                    
fprintf('-- Randomly initialize theta of the softmax classifier...\n');
% Randomly initialize the parameters
numClasses = 12;
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL1 * numClasses, 1);

% Train the softmax classifier, the classifier takes in input of dimension 
% "hiddenSizeL2" corresponding to the hidden layer size of the 2nd layer.
% You should store the optimal parameters in saeSoftmaxOptTheta

fprintf('-- Train the softmax classifier based on the data and train labels given...\n');
softmaxModel = softmaxTrain(hiddenSizeL1, numClasses, lambda, ...
                            sae1Features, trainLabels, options);
                        
saeSoftmaxOptTheta = softmaxModel.optTheta(:);

%-------------------------------------
%% Step 6: Fine-tune the whole network
%  Fine-tune the whole network by backpropate the whole network.
fprintf('\nPART 6: FINE-TUNE THE WHOLE NETWORK\n');
fprintf('===================================\n');
addpath 'Softmax Classifer'
% Implement the stackedAECost to give the combined cost of the whole model
% then run this cell.

fprintf('-- Initialize the weight stack of all three layers...\n');
% Initialize the stack using the parameters learned
stack = cell(1,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
% StackedAEtheta is the weights of the softmax concantenated with all
% weights of two other hidden layers.
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

fprintf('-- Finetuning the whole weight stack of all two layers...\n');
stackedAEOptTheta = stackedAETheta;

tic;
[stackedAEOptTheta, cost] = minFunc( @(p)stackedAECost(p, inputSize, hiddenSizeL1, ...
                                              numClasses, netconfig, ...
                                              lambda, trainData, trainLabels), ...
                              stackedAEOptTheta, options);
toc;
%%-------------
%% Step 7: Test
% Using the new trained system to predict protein subcellular localization
% in the testing set.
fprintf('\nPART 6: TEST THE WHOLE NETWORK\n');
fprintf('==============================\n');
fprintf('-- Initializing some useful variables...\n');
% Initialize the testing size and the prediction labels
numTest = size(testProteins, 2);
predictedLabels = zeros(1,numTest);
fprintf('-- Testing based on the weights before fine-tuning...\n');
% Predict localization type with weights before fine-tuning (StackedAETheta)
for proteinNum = 1:numTest
    protein = testProteins(:,proteinNum);
    predictedLabels(proteinNum) = predictType(stackedAETheta, inputSize, hiddenSizeL1, ...
                          numClasses, netconfig, protein, windowSize);
end
acc = mean(testLabels(:) == predictedLabels(:));
fprintf('   Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

fprintf('-- Testing based on the weights after fine-tuning...\n');
% Predict localization type with weights after fine-tuning (StackedOptAETheta)
for proteinNum = 1:numTest
    protein = testProteins(:,proteinNum);
    predictedLabels(proteinNum) = predictType(stackedAEOptTheta, inputSize, hiddenSizeL1, ...
                          numClasses, netconfig, protein, windowSize);
end
acc = mean(testLabels(:) == predictedLabels(:));
fprintf('   After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);