%% Using Convolutional Neural Network to predict Protein Secondary Strucutre
%  This method extracts protein substring using sliding window method and
%  then use Convolutional Neural Network to classify protein secondary
%  structure. The network is as followed
%  {input: 21 x 15} -> {Filter: 21 x 4 x 30} -> {Convolution: 30 x 12} ->
%  {Pooling 30 x 6} -> {Softmax classification}

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
trainingSize = 400;
fprintf('-- Randomly pick proteins into training set and testing set...\n');
[trainProteins, testProteins] = pickRandomProteins(proteinTable,trainingSize);
fprintf('   # of training proteins: %d\n', trainingSize);
fprintf('   # of testing proteins: %d\n', numProteins - trainingSize);
% It is super highly likely that each protein already contains all 3
% localization type, there is no need to test for labels. This could be a
% TODO.

% Slide protein into substring of some arbitrary size
windowDim = 15;
filterDim = 4;
windowSize = windowDim * 21;
sampleSize = filterDim * 21;

trainProteins = lengthenProteins(trainProteins,windowDim);
testProteins = lengthenProteins(testProteins,windowDim);

% Inject noise to training proteins. The noise injection helps the system
% find better features.
noisePercentage = 0.50;
noiseProteins = injectNoise(trainProteins,noisePercentage);

fprintf('-- Sliding windows into substring of size %d...\n', windowDim);
sampleWindows = getSamples(trainProteins,filterDim);
noisyWindows = getSamples(noiseProteins,filterDim);

% Convert the new training data into binary data with label attached
fprintf('-- Converting sample data into binary data...\n');
fprintf('   Each sample string will be represent by %d x 21 = %d binary values\n',filterDim,sampleSize);
noisyData = convertSamples(noisyWindows, filterDim);
sampleData = convertSamples(sampleWindows, filterDim);

%---------------------------------------------
%% Step 3: Detect first-level protein features
%  Detect new features out of the set of training data using the sparse
%  autoencoder
fprintf('\nPART 3: DETECT PROTEIN FEATURES\n');
fprintf('=============================\n');
addpath 'Sparse Autoencoder'

%  Set up some useful variables
fprintf('-- Initializing useful parameters...\n');

inputSize = sampleSize;     % number of input units 
numFilters = 40;       % number of hidden units of layer 1
sparsityParam = 0.20;      % desired average activation of the hidden units.
lambda = 0.0001;          % weight decay parameter       
beta = 1;                 % weight of sparsity penalty term

%  Randomly initialize the parameters
saeTheta = initializeParameters(numFilters, inputSize);

% Set sae1OptTheta equals to the initialialized sae1Theta, this will become
% our trained parameters.
saeOptTheta = saeTheta;

%  Use minFunc to minimize the function
fprintf('-- Detecting features...\n');
addpath 'minFunc'
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 1000;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';



tic;
[saeOptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, numFilters, ...
                                   lambda, sparsityParam, ...
                                   beta, noisyData, sampleData), ...
                              saeTheta, options);
toc;

% Visualizing found features.
Wc = reshape(saeOptTheta(1:numFilters*inputSize), numFilters, inputSize);
bc = saeOptTheta(2*numFilters*inputSize+1:2*numFilters*inputSize+numFilters);

figure(1);
displayProteinFeatures(Wc', 1, 1, 20, 0, filterDim, 21);
title('First-level features')
set(gcf,'name','window title string','numbertitle','off')


%---------------------------------------------------------------
%% Step 4: Convolve the found features with current training data
fprintf('\nPART 4: CONVOLVE FOUND FEATURES WITH TRAINING DATA\n');
fprintf('==================================================\n');

fprintf('-- Converting proteins from letter form to binary form...\n');
% convert filters to 21 x sampleLength image form
Wc = reshape(Wc,21,filterDim,numFilters);

trainProteins = convertProteinsToBinary(trainProteins);

fprintf('-- Convolved the protein with found filters...\n');
convolvedTrainProteins = convolveProteins(trainProteins, Wc);

fprintf('-- Slide proteins into smaller substring. These will be "convolved strings"...\n');
[trainData,trainLabels] = slideConvolvedProteins(convolvedTrainProteins, windowDim, filterDim, numFilters);
trainLabels = remapLabels(trainLabels);

%-------------------------------------
%% Step 5: Train the softmax classifer
%  This trains the softmax classifier for determining the final labels
fprintf('\nPART 5: TRAIN THE SOFTMAX CLASSIFIER\n');
fprintf('====================================\n');
addpath 'Softmax Classifier';

fprintf('-- Randomly initialize theta of the softmax classifier...\n');
numClasses = 3;
convolvedLength = windowDim - filterDim + 1;
hiddenSize = numFilters * convolvedLength;
saeSoftmaxTheta = 0.005 * randn(hiddenSize * numClasses, 1);

fprintf('-- Train the softmax classifier based on the data and train labels given...\n');
softmaxModel = softmaxTrain(hiddenSize, numClasses, lambda, ...
                            trainData, trainLabels, options);
                        
saeSoftmaxOptTheta = softmaxModel.optTheta(:);

% Extract Wd and bd from saeSoftmaxOptTheta
Wd = reshape(saeSoftmaxOptTheta(1 : hiddenSize * numClasses), numClasses, hiddenSize);

% Cluster Wc, Wd, bc, bd into the same theta;
theta = [Wc(:);Wd(:);bc(:)];

%----------------------------------------------------------
%% Step 6: Fine-tune the whole Convolutional Neural Network
% Using stochastic gradient descent to minimize the the overall cost of the
% WHOLE network

% Set up finetuneData and finetuneLabels for finetuning the whole network
%[finetuneData, finetuneLabels] = slideProteins( trainProteins, windowDim );
%finetuneLabels = remapLabels(finetuneLabels);

% Set up some options parameter
%options.epochs = 3;
%options.minibatch = 800;
%options.alpha = 1e-1;
%options.momentum = .95;
%lambda = 0.0001;

% Use stochastic gradient descent to train the WHOLE network
%opttheta = minFuncSGD(@(x,y,z) cnnCost(x, lambda, windowDim, filterDim, y,z, numClasses,numFilters),theta,finetuneData,finetuneLabels,options);

%options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
%options.maxIter = 1000;	  % Maximum number of iterations of L-BFGS to run 
%options.display = 'on';

%tic;
%[theta, cost] = minFunc( @(x) cnnCost(x, lambda, windowDim, filterDim, trainData, trainLabels, numClasses, numFilters), theta, options);
%toc;

%cnnCost(theta, lambda, windowDim, filterDim,...
%                                                trainData, trainLabels, ...
%                                                numClasses, numFilters, pred)

%--------------
%% Step 7: Test
% Using the new trained system to predict the digits in the testing set
fprintf('\nPART 7: TEST THE NEW SYSTEM\n');
fprintf('===========================\n');
fprintf('-- Testing on the training set...\n');
% Initialize labels and predictions (for decision matrix later)
pred = softmaxPredict(softmaxModel,trainData);
numAccurate = sum(pred == trainLabels);
numWindows = size(trainLabels,2);
% Calculate accuracy
acc = numAccurate / numWindows;
fprintf('   Accuracy on training set: %0.3f%%\n', acc * 100);

% Create a confusion matrix
[confusionMatrix,label] = confusionmat(trainLabels,pred);

confusionMatrix = bsxfun(@rdivide, confusionMatrix .* 100, sum(confusionMatrix,2));
Row = {'C';'E';'H'};
C_pred = confusionMatrix(:,1);
E_pred = confusionMatrix(:,2);
H_pred = confusionMatrix(:,3);
confusionMatrix = table(C_pred,E_pred,H_pred,...
    'RowNames',Row)

fprintf('-- Testing on the testing set...\n');
% Initialize predictions for the testing set
numWindows = 0;
numAccurate = 0;
% Initialize labels and predictions (for decision matrix later)
allAccurate = zeros(1,0);
allPred = zeros(1,0);
testProteins = convertProteinsToBinary(testProteins);
convolvedTestProteins = convolveProteins(testProteins, Wc);
numProteins = size(testProteins,2);
% Make topology prediction for each protein
for proteinNum = 1:numProteins
    [testData,testLabels] = slideConvolvedProteins(convolvedTestProteins(:,proteinNum), windowDim, filterDim, numFilters);
    testLabels = remapLabels(testLabels);
    % Make prediction
    pred = softmaxPredict(softmaxModel,testData);
    numWindows = numWindows + size(pred,2);
    numAccurate = numAccurate + sum(pred == testLabels);
    allAccurate = [allAccurate,testLabels];
    allPred = [allPred,pred];
end

% Calculate accuracy
acc = numAccurate / numWindows;
fprintf('   Accuracy on testing set: %0.3f%%\n', acc * 100);

% Create a confusion matrix
[confusionMatrix,label] = confusionmat(allAccurate,allPred);

confusionMatrix = bsxfun(@rdivide, confusionMatrix .* 100, sum(confusionMatrix,2));
Row = {'C';'E';'H'};
C_pred = confusionMatrix(:,1);
E_pred = confusionMatrix(:,2);
H_pred = confusionMatrix(:,3);
confusionMatrix = table(C_pred,E_pred,H_pred,...
    'RowNames',Row)