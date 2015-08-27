%% Using Stacked Autoencoders to predict Protein Secondary Strucutre Prediction
%  This method extracts protein substring using sliding window method and
%  then use Stacked Autoencoders to classify protein secondary structure
%  prediction

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
addpath 'Scratch-1D Datasets';

% Put the whole protein file into a text scanner and scan all the file in
% to a cell array.
fprintf('-- Scanning the whole protein data and store it in a cell...\n');
fileID = fopen('Proteins.fa');
proteins = textscan(fileID,'%s');
fclose(fileID);
proteins = proteins{1};
% Keep only even rows (rows that contains protein string 
proteins = proteins(2:2:end);
% Tranposing the protein data (Storing in column is standard practice
proteins = proteins';

% Put the secondary structure data into a cell array
fprintf('-- Scanning the whole protein secondary strucutre data and store it in a cell...\n');
fileID = fopen('SSpro.dssp');
sslabels = textscan(fileID,'%s');
fclose(fileID);
sslabels = sslabels{1};
% Keep only even rows (rows that contains secondary strucutre data)
sslabels = sslabels(2:2:end);
% Tranposing the protein data (Storing in column is standard practice)
sslabels = sslabels';

proteinTable = [proteins;sslabels];

%% Step 2: Process the protein
%  Here we split the proteins we have into training and testing sets as
%  well as slide the windows to get the substring data.

fprintf('\nPART 2: READ THE PROTEIN FILE\n');
fprintf('=============================\n');

% Splitting proteins into training set and testing set
numProteins = size(proteinTable,2);
trainingSize = 2000;
fprintf('-- Randomly pick proteins into training set and testing set...\n');
[trainProteins, testProteins] = pickRandomProteins(proteinTable,trainingSize);
fprintf('   # of training proteins: %d\n', trainingSize);
fprintf('   # of testing proteins: %d\n', numProteins - trainingSize);
% It is super highly likely that each protein already contains all 3
% localization type, there is no need to test for labels. This could be a
% TODO.

% Inject noise to training proteins
noisePercentage = 0.25;
trainProteins = injectNoise(trainProteins,noisePercentage);

% Slide protein into substring of some arbitrary size
windowSize = 7;
dataSize = windowSize * 21;
fprintf('-- Sliding windows into substring of size %d...\n', windowSize);
trainingWindows = slideWindows(trainProteins,windowSize);

% Convert the new training data into binary data with label attached
fprintf('-- Converting string data into binary data...\n');
fprintf('   Each substring will be represent by %d x 21 = %d binary values\n',windowSize, windowSize * 21);
[trainData,trainLabels] = convertWindows(trainingWindows,windowSize);
% Remap the letter label to numeric label (to make it easier for the
% softmax classifer to do its job
trainLabels = remapLabels(trainLabels);

%---------------------------------------------
%% Step 3: Detect first-level protein features
%  Detect new features out of the set of training data using the sparse
%  autoencoder
fprintf('\nPART 3: DETECT PROTEIN FEATURES\n');
fprintf('=============================\n');
addpath 'Sparse Autoencoder'

%  Set up some useful variables
fprintf('-- Initializing useful parameters...\n');

inputSize = dataSize;     % number of input units 
hiddenSizeL1 = 100;       % number of hidden units of layer 1
sparsityParam = 0.25;      % desired average activation of the hidden units.
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
displayProteinFeatures(features1', 1, 1, 20, 0, windowSize, 21);
title('first-level features')
set(gcf,'name','window title string','numbertitle','off')

%-------------------------------------
%% Step 4: Train the softmax classifer
%  This trains the softmax classifier for determining the final labels
fprintf('\nPART 5: TRAIN THE SOFTMAX CLASSIFIER\n');
fprintf('====================================\n');
addpath 'Softmax Classifier'

% This part re-represent the original input as features found in
% sae1OptTheta
[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainData);
                                    
fprintf('-- Randomly initialize theta of the softmax classifier...\n');
% Randomly initialize the parameters
numClasses = 3;
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL1 * numClasses, 1);

% Train the softmax classifier, the classifier takes in input of dimension 
% "hiddenSizeL2" corresponding to the hidden layer size of the 2nd layer.
% You should store the optimal parameters in saeSoftmaxOptTheta

fprintf('-- Train the softmax classifier based on the data and train labels given...\n');
softmaxModel = softmaxTrain(hiddenSizeL1, numClasses, lambda, ...
                            sae1Features, trainLabels, options);
                        
saeSoftmaxOptTheta = softmaxModel.optTheta(:);

%-------------------------------------
%% Step 5: Fine-tune the whole network
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

fprintf('-- Finetuning the whole weight stack of all three layers...\n');
stackedAEOptTheta = stackedAETheta;

tic;
[stackedAEOptTheta, cost] = minFunc( @(p)stackedAECost(p, inputSize, hiddenSizeL1, ...
                                              numClasses, netconfig, ...
                                              lambda, trainData, trainLabels), ...
                              stackedAEOptTheta, options);
toc;
%%-------------
%% Step 6: Test
% Using the new trained system to predict the digits in the testing set

% Prediction for the training set

numWindows = 0;
numAccurate = 0;
% Initialize numProteins
numProteins = size(trainProteins,2);

for proteinNum = 1:numProteins
    % For each protein
    protein = trainProteins(:,proteinNum);
    % Initialize test data and test labels
    testWindows = slideWindows(protein,windowSize);
    % Convert the new training data into binary data with label attached
    [testData,testLabels] = convertWindows(testWindows,windowSize);
    testLabels = remapLabels(testLabels);
    % Make predictions
    [pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL1, ...
                          numClasses, netconfig, testData);
    numWindows = numWindows + numel(testLabels);
    numAccurate = numAccurate + sum(testLabels(:) == pred(:));
end

% Remap the letter label to numeric label (to make it easier for the
% softmax classifer to do its job

% Predict with weights before fine-tuning (StackedAETheta)
acc = numAccurate / numWindows;
fprintf('Before Finetuning Accuracy on training set: %0.3f%%\n', acc * 100);

% Initialize number of windows and number of accurate predictions
numWindows = 0;
numAccurate = 0;
allPred = zeros(1,0);
allAccurate = zeros(1,0);
% Initialize numProteins
numProteins = size(trainProteins,2);

for proteinNum = 1:numProteins
    % For each protein
    protein = trainProteins(:,proteinNum);
    % Initialize test data and test labels
    testWindows = slideWindows(protein,windowSize);
    % Convert the new training data into binary data with label attached
    [testData,testLabels] = convertWindows(testWindows,windowSize);
    testLabels = remapLabels(testLabels);
    % Make predictions
    [pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL1, ...
                          numClasses, netconfig, testData);
    % Add to the total number of predictions
    allPred = [allPred,pred];
    allAccurate = [allAccurate,testLabels];
    numWindows = numWindows + numel(testLabels);
    numAccurate = numAccurate + sum(testLabels(:) == pred(:));
end

% Create a confusion matrix
confusionMatrix = confusionmat(allAccurate,allPred);

% Predict with weights before fine-tuning (StackedAETheta)
acc = numAccurate / numWindows;
fprintf('After Finetuning Accuracy on training set: %0.3f%%\n', acc * 100);

% Initialize number of windows and number of accurate predictions
numWindows = 0;
numAccurate = 0;
% Initialize numProteins
numProteins = size(testProteins,2);

for proteinNum = 1:numProteins
    % For each protein
    protein = testProteins(:,proteinNum);
    % Initialize test data and test labels
    testWindows = slideWindows(protein,windowSize);
    % Convert the new training data into binary data with label attached
    [testData,testLabels] = convertWindows(testWindows,windowSize);
    testLabels = remapLabels(testLabels);
    % Make predictions
    [pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL1, ...
                          numClasses, netconfig, testData);
    numWindows = numWindows + numel(testLabels);
    numAccurate = numAccurate + sum(testLabels(:) == pred(:));
end

% Predict with weights before fine-tuning (StackedAETheta)
acc = numAccurate / numWindows;
fprintf('Before Finetuning Accuracy on testing set: %0.3f%%\n', acc * 100);

% Initialize number of windows and number of accurate predictions
numWindows = 0;
numAccurate = 0;
allPred = zeros(1,0);
allAccurate = zeros(1,0);
% Initialize numProteins
numProteins = size(testProteins,2);

for proteinNum = 1:numProteins
    % For each protein
    protein = testProteins(:,proteinNum);
    % Initialize test data and test labels
    testWindows = slideWindows(protein,windowSize);
    % Convert the new training data into binary data with label attached
    [testData,testLabels] = convertWindows(testWindows,windowSize);
    testLabels = remapLabels(testLabels);
    % Make predictions
    [pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL1, ...
                          numClasses, netconfig, testData);
    numWindows = numWindows + numel(testLabels);
    numAccurate = numAccurate + sum(testLabels(:) == pred(:));
    allPred = [allPred,pred];
    allAccurate = [allAccurate,testLabels];
end

% Predict with weights before fine-tuning (StackedAETheta)
acc = numAccurate / numWindows;
fprintf('After Finetuning Accuracy on testing set: %0.3f%%\n', acc * 100);

% Create a confusion matrix
[confusionMatrix,label] = confusionmat(allAccurate,allPred);

confusionMatrix = bsxfun(@rdivide, confusionMatrix .* 100, sum(confusionMatrix,2));
Row = {'C';'E';'H'};
C_pred = confusionMatrix(:,1);
E_pred = confusionMatrix(:,2);
H_pred = confusionMatrix(:,3);
confusionMatrix = table(C_pred,E_pred,H_pred,...
    'RowNames',Row)