%% GRID SEARCH FOR MLP. ORIGINAL IMBALANCED DATA

%% Dataset, exploratory data analysis and Method
%There are 6497 entries (before removal of missing values), 12 physiochemical wine quality predictors and 1
%target value – the quality of the wine. Continues values would need
%to be normalized since there are some really big numbers. 

%The data preprocessing is done in python. This include removal 
%of missing values, descriptive statistics, class regrouping into 
% binary classification, checking for class imbalance and predictors multicollinearity. 
%There are 38 missing values which is just over 0.5 % of total entries so
% rows with missing values can be removed with no major effect on data.

% After regruping in to binary classification is done there are 2 classes:
% average wine (5192 entries) and very good wine (1271 entries). 

% Target class labels: Average wine -0 ; Very good wine - 1

%The data is split in to training (70 %) and testing (30%) sets using
%‘holdout’ method.  

% To avoid collinearity 'free sulfur dioxide; (predictor) will be removed from our study. 
%% Clear all
clear all; clc;
%% LOADING TRAIN DATA

wine_train = readtable('wine_train.csv');

%% Normalisisng train attributes 
norm_type = zscore(wine_train.type)
norm_fixedacidity = zscore(wine_train.fixedAcidity)
norm_volatileacidity = zscore(wine_train.volatileAcidity)
norm_citricAcid = zscore(wine_train.citricAcid)
norm_residualSugar = zscore(wine_train.residualSugar)
norm_chlorides = zscore(wine_train.chlorides)

norm_totalSulfurDioxide = zscore(wine_train.totalSulfurDioxide)
norm_density = zscore(wine_train.density)
norm_pH = zscore(wine_train.pH)
norm_sulphates = zscore(wine_train.sulphates)
norm_alcohol = zscore(wine_train.alcohol)
%% NORMALISED TRAIN TABLE
% Create table with normalised atributes. 
norm_wine_train = table(norm_type, norm_fixedacidity, norm_volatileacidity, norm_citricAcid, norm_residualSugar, norm_chlorides, norm_totalSulfurDioxide, norm_density, norm_pH, norm_sulphates, norm_alcohol,wine_train.QualityGroup);

%% Partition of data to training set (80%) and validation set (20%)

part= cvpartition(norm_wine_train.Var12,'Holdout',0.2);
train_idx = training(part);
validation_idx = test(part);

% Training and validation datasets
train_data = norm_wine_train(train_idx,:);
validation_data = norm_wine_train(validation_idx,:);


%% Converting table to array and separating predictors from responce
trainingPredictors = table2array(train_data(:,1:11));
trainingResponse = table2array(train_data(:,12));
validationPredictors = table2array(validation_data(:,1:11));
validationResponse = table2array(validation_data(:,12));
% Transposing 
trainingPredictors = trainingPredictors';
trainingResponse = trainingResponse';
validationPredictors = validationPredictors';
validationResponse = validationResponse';
%% Parameters

Hidden_Layer_Size = [1,10,20,30,40,50,70] % Hidden neurons
Learning_Rate = [0.005, 0.01 0.05, 0.1 0.2, 0.5] % Learning rate
momentum = [0.8, 0.5, 0.2] % Momentum
epochs = [200 ,500, 1500, 2500] % Maximum number of epochs 

trainFcn = 'traingdm';  % Gradient descent with momentum backpropagation
net.layers{1}.transferFcn = 'tansig' %  hidden layer transfer function
net.layers{2}.transferFcn = 'logsig' % output layer  transfer function
input_processFcns = {'removeconstantrows','mapminmax'}; % Input Pre-Processing Functions

performFcn = 'crossentropy'; % Cost Function
divideFcn = 'dividerand';  % Divide data randomly
divideMode = 'sample';  % Divide up every sample
plotFcns = {'plotperform','plottrainstate','ploterrhist', 'plotconfusion', 'plotroc'}; 
            %...Plot Functions


%% Grid search table 

Table_MLP_NoSmote = table('Size', [0,7], ...
    'VariableTypes', {'double', 'double', ...
    'double', 'double', 'double', ...
    'double', 'double'}, ...
    'VariableNames', {'Hidden_Layer_Size','Learning_Rate', ...
    'Momentum', 'Training_Acc', 'Validation_Acc', ...
    'Epochs', 'Time'});

%% Grid search

tic; % start the timer
for hl = Hidden_Layer_Size
    for lr = Learning_Rate
        for m = momentum
            for e = epochs
            net = patternnet(hl, trainFcn);

            net.trainparam.learning_rate = lr;
            net.trainparam.momentum = m;
            net.trainparam.epochs = e;
          
            net.plotFcns = {'plotperform','plottrainstate','ploterrhist', 'plotconfusion', 'plotroc'}; 
            %...Plot Functions

            performFcn = 'crossentropy';
            net.divideFcn = 'dividerand';  
            net.divideMode = 'sample';  
            
            % Train the Network
            [net,tr] = train(net,trainingPredictors,trainingResponse);
            
            % Test the Network
           
            trainY = net(trainingPredictors);
            [c,cm] = confusion(trainingResponse,trainY);
            %Train Accuracy = (TP+TN)/(TP+FP+FN+TN)
            Train_Acc =  100*(1-c);
          
            ValY = net(validationPredictors);
            [b,bm] = confusion(validationResponse,ValY);
            % Validation Accuracy = (TP+TN)/(TP+FP+FN+TN)
            Val_Acc =  100*(1-b);

            Time = toc;  % stop the timer   
            names = {hl,lr,m,Train_Acc, Val_Acc, e, Time};
            Table_MLP_NoSmote = [Table_MLP_NoSmote; names];
            
            end
        end
    end
end