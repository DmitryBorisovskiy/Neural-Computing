%% GRID SEARCH FOR SVM. DATA IS REBALANCED BY BORDERLINE SMOTE

%% Dataset, exploratory data analysis and Method
%There are 6497 entries (before removal of missing values and SMOTE), 12 physiochemical wine quality predictors and 1
%target value – the quality of the wine. Continues values would need
%to be normalized since there are some really big numbers. 

%The data preprocessing is done in python. This include removal 
%of missing values, descriptive statistics, class regrouping into 
% binary classification, checking for class imbalance and predictors multicollinearity. 
%There are 38 missing values which is just over 0.5 % of total entries so
% rows with missing values can be removed with no major effect on data.

% After regruping in to binary classification is done there are 2 classes:
% average wine (5192 entries) and very good wine (1271 entries).  Borderline SMOTE 
% is applied to training set to resolve the issue with class imbalance.

% Target class labels: Average wine -0 ; Very good wine - 1

%The data is split in to training (70 %) and testing (30%) sets using
%‘holdout’ method.  

% To avoid collinearity 'free sulfur dioxide; (predictor) will be removed from our study. 

%% Clear all
clear all; clc;
%% Load Train data

wine_train_smote = readtable('wine_train_smote.csv');

%% Shuffle observations - shuffle was added to avoid overfitting and reduce variance 

wine_train_smote = wine_train_smote(randperm(size(wine_train_smote,1)),:);
%% Normalisisng the attributes 
norm_type = zscore(wine_train_smote.type);
norm_fixedacidity = zscore(wine_train_smote.fixedAcidity);
norm_volatileacidity = zscore(wine_train_smote.volatileAcidity);
norm_citricAcid = zscore(wine_train_smote.citricAcid);
norm_residualSugar = zscore(wine_train_smote.residualSugar);
norm_chlorides = zscore(wine_train_smote.chlorides);

norm_totalSulfurDioxide = zscore(wine_train_smote.totalSulfurDioxide);
norm_density = zscore(wine_train_smote.density);
norm_pH = zscore(wine_train_smote.pH);
norm_sulphates = zscore(wine_train_smote.sulphates);
norm_alcohol = zscore(wine_train_smote.alcohol);

%% NORMALISED TABLE
% Create table with normalised atributes. 
norm_wine_train_smote = table(norm_type, norm_fixedacidity, norm_volatileacidity, norm_citricAcid, norm_residualSugar, norm_chlorides, norm_totalSulfurDioxide, norm_density, norm_pH, norm_sulphates, norm_alcohol,wine_train_smote.QualityGroup);

%% Separating predictors from responce
predictorNames = {'norm_type', 'norm_fixedacidity', 'norm_volatileacidity', 'norm_citricAcid', 'norm_residualSugar', 'norm_chlorides', 'norm_totalSulfurDioxide', 'norm_density', 'norm_pH', 'norm_sulphates', 'norm_alcohol'};
predictors = norm_wine_train_smote(:, predictorNames);
response = norm_wine_train_smote.Var12;
%% Partition of data to training set (80%) and validation set (20%)
% Set up holdout validation
cvp = cvpartition(response, 'Holdout', 0.2);
trainingPredictors = predictors(cvp.training, :);
trainingResponse = response(cvp.training, :);
% Validation predictions
validationPredictors = predictors(cvp.test, :);
validationResponse = response(cvp.test, :);


%% Hypoparameters
% Kernel Function :linear, rbf, polynomial
 BoxConstant = [0.005, 0.01, 0.03, 0.05]; % Box Constant for all kernel functions
 Polynomial_function_order = [1,2,3,4,5]; % For polynomial kernel function
%% Grid search 1 - Kernel Function: Linear
 
SVM_Table_linear = table('Size', [0,3], ...
    'VariableTypes', {'double', 'double', 'double'}, ...
    'VariableNames', {'BoxConstant','Validation_Acc', 'Time'});
%%
tic % start the timer
for BC = BoxConstant
    classificationSVM_linear = fitcsvm(trainingPredictors, trainingResponse,'KernelFunction', 'linear',...
    'BoxConstraint',BC);
    p_linear = predict(classificationSVM_linear, validationPredictors);
    accuracy_linear = sum(p_linear == validationResponse) / length(validationResponse)*100
    Time = toc; % stop the timer 
     hrow_linear= {BC,accuracy_linear, Time};
     SVM_Table_linear = [SVM_Table_linear; hrow_linear];
end

%% Grid search 2 - Kernel Function: RBF
SVM_Table_rbf = table('Size', [0,3], ...
    'VariableTypes', {'double', 'double', 'double'}, ...
    'VariableNames', {'BoxConstant','Validation_Acc', 'Time'});
%%
tic % start the timer
for BC = BoxConstant
    classificationSVM_rbf = fitcsvm(trainingPredictors, trainingResponse,'KernelFunction', 'rbf',...
    'BoxConstraint',BC);
    p_rbf = predict(classificationSVM_rbf, validationPredictors);
    accuracy_rbf = sum(p_rbf == validationResponse) / length(validationResponse)*100
     Time = toc; % stop the timer 
     hrow_rbf= {BC,accuracy_rbf, Time};
     SVM_Table_rbf = [SVM_Table_rbf; hrow_rbf];
end
%% Grid search 3 - KernelFunction: Polynomial

SVM_Table_polynomial = table('Size', [0,4], ...
    'VariableTypes', {'double', 'double','double', 'double'}, ...
    'VariableNames', {'BoxConstant','Validation_Acc', 'Polynomial_function_order', 'Time' });
%%
tic % start the timer
for BC = BoxConstant
    for PO = Polynomial_function_order
    classificationSVM_polynomial = fitcsvm(trainingPredictors, trainingResponse,'KernelFunction', 'polynomial',...
    'BoxConstraint',BC, 'PolynomialOrder', PO);
    p_polynomial = predict(classificationSVM_polynomial, validationPredictors);
    accuracy_polynomial = sum(p_polynomial == validationResponse) / length(validationResponse)*100
    Time = toc; % stop the timer 
    hrow_polynomial = {BC,accuracy_polynomial,PO,Time };
    SVM_Table_polynomial = [SVM_Table_polynomial; hrow_polynomial];
    end
end
       