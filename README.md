# Neural-Computing
A comparative study between Multilayer Perceptron and Support Vector Machine applied to predicting quality of wine

The following files are included: 

• BEST_MODELS_FINAL.m – Matlab Code: contains best SVM and MLP models
trained with the best hyperparameters on original (no SMOTE) data and tested on test
data.
• MLP_NoSmote.m - Matlab Code: Grid Search for MLP on original data
• MLP_SMOTE.m - Matlab Code: Grid Search for MLP on SMOTE rebalanced data.
• SVM_NoSmote.m - Matlab Code Grid Search for SVM on original data.
• SVM_Smote.m - Matlab Code: Grid Search for SVM on SMOTE rebalanced data.
• Data_Preprocessing .ipynb - ipynb File: Data preprocessing which include removal
of missing values, descriptive statistics, class regrouping into binary classification,
checking for class imbalance and predictors multicollinearity. The data is split in to
training (70 %) and testing (30%) sets using ‘holdout’ method. Borderline SMOTE is
applied to training set to resolve the issue with class imbalance.
Report – PDF File
