%% BEST MODELS COMPARISON: MLP VS SVM 
% Following the grid search for both algorithms the presense of the following
% hypeparameters showed the best accuracy results: 

% Best MLP Hyperparameters:
% Hidden Layer Size = 30;
% Learning rate = 0.2;
% Momentum = 0.5;
% Epochs = 2500;

% Best SVM Hyperparameters:
% Kernel Function = Polynomial
% BoxConstant = 0.005
% Polynomial function order = 4

%In the testing process it has been decided to keep the original imbalance 
%dataset to train the best MLP and SVM models using hyperparameters above

%% Dataset, exploratory data analysis and Method
%There are 6497 entries (before removal of missing values), 12 physiochemical wine quality predictors and 1
%target value – the quality of the wine. Continues values would need
%to be normalized since there are some really big numbers. 

%The data preprocessing is done in python. This include removal 
%of missing values, descriptive statistics, class regrouping into 
% binary classification, checking for class imbalance and predictors multicollinearity. 
%There are 38 missing values which is just over 0.5 % of total entries so
% rows with missing values can be removed with no major effect on data.
 
%After regruping in to binary classification is done there are 2 classes:
% average wine (5192 entries) and very good wine (1271 entries).

% Target class labels: Average wine -0 ; Very good wine - 1

%The data is split in to training (70 %) and testing (30%) sets using
%‘holdout’ method.  

% To avoid collinearity 'free sulfur dioxide; (predictor) will be removed from our study. 

%% Clear all
clear all; 
clc;
%% LOADING TRAIN DATA

wine_train = readtable('wine_train.csv');


%% Normalisisng train attributes 
norm_type_train = zscore(wine_train.type);
norm_fixedacidity_train = zscore(wine_train.fixedAcidity);
norm_volatileacidity_train = zscore(wine_train.volatileAcidity);
norm_citricAcid_train = zscore(wine_train.citricAcid);
norm_residualSugar_train = zscore(wine_train.residualSugar);
norm_chlorides_train = zscore(wine_train.chlorides);

norm_totalSulfurDioxide_train = zscore(wine_train.totalSulfurDioxide);
norm_density_train = zscore(wine_train.density);
norm_pH_train = zscore(wine_train.pH);
norm_sulphates_train = zscore(wine_train.sulphates);
norm_alcohol_train = zscore(wine_train.alcohol);

%% NORMALISED TRAIN TABLE
% Create table with normalised atributes. 
norm_wine_train = table(norm_type_train, norm_fixedacidity_train, norm_volatileacidity_train, norm_citricAcid_train, norm_residualSugar_train, norm_chlorides_train, norm_totalSulfurDioxide_train, norm_density_train, norm_pH_train, norm_sulphates_train, norm_alcohol_train,wine_train.QualityGroup);

%% Converting table to array (Train) and separating predictors from responce
predictors_train = table2array(norm_wine_train(:,1:11));
response_train = table2array(norm_wine_train(:,12));
%% Transposing predictors and response for MLP model
tr_predictors_train = predictors_train';
tr_response_train = response_train';


%% LOADING TEST DATA

wine_test = readtable('wine_test.csv');

%% Normalisisng test attributes 

norm_type_test = zscore(wine_test.type);
norm_fixedacidity_test = zscore(wine_test.fixedAcidity);
norm_volatileacidity_test = zscore(wine_test.volatileAcidity);
norm_citricAcid_test = zscore(wine_test.citricAcid);
norm_residualSugar_test = zscore(wine_test.residualSugar);
norm_chlorides_test = zscore(wine_test.chlorides);

norm_totalSulfurDioxide_test = zscore(wine_test.totalSulfurDioxide);
norm_density_test = zscore(wine_test.density);
norm_pH_test = zscore(wine_test.pH);
norm_sulphates_test = zscore(wine_test.sulphates);
norm_alcohol_test = zscore(wine_test.alcohol);

%% NORMALISED TEST TABLE
% Create table with normalised atributes. 
norm_wine_test = table(norm_type_test, norm_fixedacidity_test, norm_volatileacidity_test, norm_citricAcid_test, norm_residualSugar_test, norm_chlorides_test, norm_totalSulfurDioxide_test, norm_density_test, norm_pH_test, norm_sulphates_test, norm_alcohol_test,wine_test.QualityGroup);
%% Converting table to array (Test)  and separating predictors from responce
predictors_test = table2array(norm_wine_test(:,1:11));
response_test = table2array(norm_wine_test(:,12));
%% Transposing predictors and response for MLP model
tr_predictors_test = predictors_test';
tr_response_test = response_test';

%% BUILDING BEST MLP MODEL

% Best MLP Hyperparameters:
Hidden_Layer_Size_best = 30; % Hidden neurons
Learning_rate_best = 0.2; % Learning rate
Momentum_best = 0.5; % Momentum
Epochs_best = 2500; % Maximum number of epochs 
Training_Function = 'traingdm' % Gradient descent with momentum backpropagation
%% Best MLP Model

net = patternnet(Hidden_Layer_Size_best, Training_Function,'crossentropy');

net.trainparam.learning_rate = Learning_rate_best;
net.trainparam.momentum = Momentum_best;
net.trainparam.epochs = Epochs_best
            
net.layers{1}.transferFcn = 'tansig' %  hidden layer transfer function
net.layers{2}.transferFcn = 'logsig' % output layer  transfer function
input_processFcns = {'removeconstantrows','mapminmax'}; % Input Pre-Processing Functions
performFcn = 'crossentropy'; % Cost Function
net.divideFcn = 'dividerand'; % Divide data randomly
net.divideMode = 'sample';  %  Divide up every sample
                   
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', 'plotconfusion', 'plotroc'}; 
            %...Plot Functions
                     
% Train the Network
[net,tr] = train (net,tr_predictors_train,tr_response_train);
                        
% Test the Network
 Predict = net(tr_predictors_test);
                       
%Test Accuracy
[c,cm] = confusion(tr_response_test,Predict);
MLP_Accuracy =  100*(1-c);

%% Confusion Matrix for MLP (test data)
plotconfusion_MLP = plotconfusion(tr_response_test,Predict);
title('MLP Test Confusion Matrix ')

%% Precision, Recall and F1 Score for MLP 

% Precision = Precision = TP/(TP+FP)
MLP_precision = cm(1:1)/(sum(cm(:,1)))
% Recall = TP/(TP+FN)
MLP_recall = cm(1:1)/(sum(cm(1,:)))
% F1 Score = 2*(Recall * Precision) / (Recall + Precision)
MLP_F1 = 2*(MLP_precision*MLP_recall)/(MLP_recall + MLP_precision)


%% BUILDING BEST SVM MODEL

% Best SVM Hyperparameters:
Kernel_Function = 'polynomial';
Box_Constant = 0.005;
Polynomial_function_order = 4;


%% Best SVM Model
Best_SVM_classification = fitcsvm(predictors_train, response_train,'KernelFunction', 'polynomial',...
'BoxConstraint',0.005, 'PolynomialOrder', 4);
Predict_SVM_Best = predict(Best_SVM_classification, predictors_test);
% SVM accuracy = (TP+TN)/(TP+FP+FN+TN)
SVM_Accuracy = sum(Predict_SVM_Best == response_test) / length(response_test)*100

%% Confusion Matrix for SVM (test data)
tr_Predict_SVM_Best = Predict_SVM_Best'
plotconfusion_SVM = plotconfusion(tr_response_test,tr_Predict_SVM_Best)
title('SVM Test Confusion Matrix ')

%% Precision, Recall and F1 Score for SVM 
mm = confusionmat(response_test,Predict_SVM_Best);
% Precision = TP/(TP+FP)
SVM_precision = sum(mm(1:1))/sum(mm(1,:)) ;
% Recall = TP/(TP+FN)
SVM_recall = sum(mm(1:1))/sum(mm(:,1));
% F1 Score = 2*(Recall * Precision) / (Recall + Precision)
SVM_F1 = 2*(SVM_precision*SVM_recall)/(SVM_recall + SVM_precision);

%% Test ROC curve for SVM and MLP

[labels_svm,score_svm] = predict(Best_SVM_classification, predictors_test );
[Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(response_test,score_svm(:,2),1);
[Xmlp,Ymlp,Tmlp,AUCmlp] = perfcurve(tr_response_test,Predict,1);
%% Plotting ROC curves
plot(Xsvm,Ysvm)
hold on
plot(Xmlp,Ymlp)
legend('SVM','MLP')
xlabel('False positive rate') 
ylabel('True positive rate')
title('Test ROC Curves for SVM and MLP Classification')
hold off
