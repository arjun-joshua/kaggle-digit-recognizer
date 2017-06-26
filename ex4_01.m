%% Machine Learning Online Class - Exercise 4 Neural Network Learning
% MODIFIED FOR KAGGLE DIGIT RECOGNIZER. 1 HIDDEN LAYER

%% Initialization
% % clear; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 784;  % 28x28 Input Images of Digits; Coursera size was 400
hidden_layer_size = 100;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============

% Load Training Data
% % X = csvread('D:\temp\kaggle digit recognizer data\train.csv',1,0);
% % y = X(:,1); X(:,1) = []; %labels are stored in the first column, images in the rest
% % y(y==0) = 10;% replace 0 with 10 so as to be able to use y as an indexing array

% Load Test Data
% % Xunlabeled = csvread('D:\temp\kaggle digit recognizer data\test.csv',1,0);

% Load previously calculated weights for network archetecture defined above
% % load('submission08.mat');
initial_Theta1 = [];%Theta1; 
initial_Theta2 = [];%Theta2;

% Randomize order of digits in training data
% % sel = randperm(size(X, 1));
% % X = X(sel,:); y = y(sel);

% IMAGE PREPROCESSING
Xmod = X; XunlabeledMod = Xunlabeled;
[Xmod, mu, sigma] = featureNormalize(Xmod');Xmod = Xmod';mu = mu';sigma = sigma';%Xmod = Xmod/255;    
[XunlabeledMod, muUnlabeled, sigmaUnlabeled] = featureNormalize(XunlabeledMod');XunlabeledMod = XunlabeledMod';muUnlabeled = muUnlabeled';sigmaUnlabeled = sigmaUnlabeled';%XunlabeledMod = XunlabeledMod/255;  
% % % Add contrast. Multiplication factor is contrast control.
% % Xmod = tanh((Xmod - 0.5) * 5);
% % XunlabeledMod = tanh((XunlabeledMod - 0.5) * 5);

%%%Randomly select 100 data points to display
% % sel = sel(1:100);
% % 
% % displayData(Xmod(sel, :));

mFull = size(X, 1);
Xtrain = Xmod(1:0.8*mFull,:); yTrain = y(1:0.8*mFull,:);%Xtrain = Xmod(1:0.6*mFull,:); yTrain = y(1:0.6*mFull,:);
Xval = Xmod(0.8*mFull+1:end,:); yVal = y(0.8*mFull+1:end,:);%Xval = Xmod(0.6*mFull+1:0.8*mFull,:); yVal = y(0.6*mFull+1:0.8*mFull,:);
% Xtest = Xmod(0.8*mFull+1:end,:); yTest = y(0.8*mFull+1:end,:);


%% =================== Part 8: Training NN ===================
%
fprintf('\nTraining Neural Network... \n')

%  You should also try different values of lambda
lambda = 1.4;
nIter = 50; %number of iterations in fmincg options in learningCurve.m

[errorTrain,errorVal, Theta1, Theta2] = learningCurve(Xtrain, yTrain, Xval, yVal, ...
                                        input_layer_size, hidden_layer_size, num_labels, lambda, ...
                                        nIter, initial_Theta1, initial_Theta2);

% PLOT LEARNING CURVES                                    
% % figure;clf;
% % plot(1:24,errorTrain,1:24, errorVal)
% % title({strcat('Learning curve for NN with \lambda = ',num2str(lambda),', n\_iteration = ',num2str(nIter)), strcat('n\_input = ',...
% % num2str(input_layer_size),', n\_hidden = ',num2str(hidden_layer_size),...
% % ', n\_labels = ',num2str(num_labels))})
% % legend('Train', 'Cross Validation')
% % xlabel('Number of training examples')
% % ylabel('Error')
%% ================= Part 9: Visualize Weights =================
%  You can now "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

% displayData(Theta1(:, 2:end));

%% ================= Part 10: Implement Predict =================

% % predTest = predict(Theta1, Theta2, Xtest);

%%%%prediction on Kaggle's test data exported as csv
% % predUnlabeled = predict(Theta1, Theta2, XunlabeledMod);
% % predUnlabeled(predUnlabeled==10) = 0;%replace label 10 with label 0 for Kaggle submission
% % csvwrite('submission08_01.csv',[ ( 1:size(XunlabeledMod,1) )' predUnlabeled ]);

fprintf('\nTraining Set Accuracy: %f\n', 100-errorTrain(end));
fprintf('\nCross Validation Set Accuracy: %f\n', 100-errorVal(end));
% % fprintf('\nTest Set Accuracy: %f\n', mean(double(predTest == yTest)) * 100);

