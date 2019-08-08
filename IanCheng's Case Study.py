import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble,  gaussian_process
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.metrics import classification_report_imbalanced


# Set working directory
os.chdir(r"D:\STUDY")
# Read the raw data, delimiter="\t"
data = pd.read_csv("DScasestudy.txt", delimiter="\t")
# Set dependent and independent variables
y = data["response"]
X = data.drop("response", axis=1)
# Count the quantity by category
print(pd.value_counts(y))
# Check for the missing values
print('Dataset columns with null values:\n', data.columns[data.isnull().any()])

# Set the number of principal components of the principal component model
pca = PCA(n_components=5)
# Fit the pca model
pca.fit(X)
# Convert raw data
X_pca = pca.fit_transform(X)
# Split the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=1/3, random_state=1)
# Specify model name
model_name = ["AdaBoostClassifier", "BaggingClassifier", "ExtraTreesClassifier",
              "GradientBoostingClassifier", "RandomForestClassifier", "GaussianProcessClassifier",
              "Logistic Regression", "BernoulliNB", "Naives Bayes", "KNeighborsClassifier", 
              "Support Vector Machine", "NuSVC", "LinearSVC", "Decision Tree",
              "XGBoost"]
# Specify a list of models
model_list = [
        #Ensemble Methods
        ensemble.AdaBoostClassifier(),
        ensemble.BaggingClassifier(),
        ensemble.ExtraTreesClassifier(),
        ensemble.GradientBoostingClassifier(),
        ensemble.RandomForestClassifier(n_estimators=100, random_state=1),

        #Gaussian Processes
        gaussian_process.GaussianProcessClassifier(),
        
        #GLM
        linear_model.LogisticRegression(solver="liblinear", random_state=1),
        linear_model.RidgeClassifierCV(), 
    
        #Navies Bayes
        naive_bayes.BernoulliNB(),
        naive_bayes.GaussianNB(),
    
        #Nearest Neighbor
        neighbors.KNeighborsClassifier(),
    
        #SVM
        svm.SVC(gamma="auto", random_state=1),
        svm.NuSVC(probability=True),
        svm.LinearSVC(),
    
        #Trees    
        tree.DecisionTreeClassifier(random_state=1),   

        #xgboost
        XGBClassifier()]

print("Dealing with unbalanced data")
# Smote method
sm = SMOTE(random_state=1)
# Resampling data
X_train_res, y_train_res = sm.fit_sample(X_pca, y)
# Count the different counts
print(pd.value_counts(y_train_res))
for name, model in zip(model_name, model_list):
    # The model was fitted according to the training set and the test set
    model.fit(X_train_res, y_train_res)
    # Cross validation using the training set and the test set after smote
    val_score = cross_val_score(model, X_train_res, y_train_res, cv=5)
    mean_score = np.round(np.mean(val_score), 4)
    print("5 fold val score of model %s is %s" % (name, mean_score))
    y_test_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_test_pred))
    # precision, recall, specificity, geometric mean, f1 score and index balanced accuracy of the geometric mean
    print(classification_report_imbalanced(y_test, y_test_pred))
    print("\n")




