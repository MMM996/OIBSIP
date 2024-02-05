# This is the code for Task 1 Oasis byte internship
# Task 1: IRIS Flower Classification
# The Code is written by Muhammad Mudassir Majeed
# The date is Jan-24.
# Dataset: https://www.kaggle.com/datasets/saurabh00007/iriscsv

#-----------------------------------------------------------------------------#

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Part 1: Load Dataset
data = pd.read_csv('Task 1\iris.csv')

# Part 2: Preliminary EDA
data.head()
data.columns
data.info()
data.describe()

# Check for Null Values
data.isnull().sum()     # No Null Values

# Check for Duplicate Values
data.duplicated().sum()     # No Duplicates

# Count Unique Values for Class Columns
data['Species'].nunique()   # 3 Unique classes. Can use label encoding or 1 hot encoding

# Observe Class Balance
data['Species'].value_counts()
ax = sns.countplot(x = data['Species'], label = 'Count')
# Equal class balances. No pre-processing required in this regard

# Observe Variable Distributions
grid_row, grid_col = 3,3
plt.figure(figsize=(12,10))
plt.suptitle('Histograms for all features')

for i,column in enumerate(data.columns,1):
    plt.subplot(grid_row, grid_col, i)
    sns.histplot(data = data[column], kde=True)
    plt.title('Histogram of {}'.format(column))
    
plt.tight_layout(rect=[0, 0.03, 1 , 0.99])
plt.show()


# Part 3: Pre-processing

# Drop Unnecessary Columns
data = data.drop(['Id'], axis =1)

# Split Data into X and Y
X = data.drop(['Species'], axis = 1)
Y = data['Species']

# Convert Classes to Numeric
from sklearn.preprocessing import LabelEncoder

label_Encoder = LabelEncoder()
Y = label_Encoder.fit_transform(Y)

# Observe Outliers using Box Plots
grid_row, grid_col = 3,3
plt.figure(figsize=(12,10))
plt.suptitle('Box Plots for all features')

for i, column in enumerate(X.columns,1):
    plt.subplot(grid_row, grid_col, i)
    sns.boxplot(data = data[column])
    plt.title('Box Plot of {}'.format(column))
    
plt.tight_layout(rect = [0,0.03,1,0.99])
plt.show()

# Standardize
X.describe()
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_standardized = scalar.fit_transform(X)
X_stand = pd.DataFrame(data = X_standardized, columns = X.columns)
X_stand.describe()

# Observe Correlations and heatmap
f, ax = plt.subplots(figsize=(10,10))
sns.heatmap(X.corr(), annot = True, linewidth=.5, fmt='.1f', ax=ax)

# Part 4: Model Training
# Split Data into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_stand, Y, test_size= 0.2, random_state=42,
                                                    stratify=Y)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
logistic.fit(X_train, Y_train)

Y_predict_logistic = logistic.predict(X_test)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, Y_train)

Y_predict_rf = rf.predict(X_test)

# AdaBoost
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# Create a base classifier like Decision Tree
base_classifier = DecisionTreeClassifier(max_depth=2)

ada = AdaBoostClassifier(base_classifier, n_estimators=100)
ada.fit(X_train,Y_train)

Y_predict_ada = ada.predict(X_test)

# XG Boost
import xgboost as xgb
# Not in sklearn. Installed using pip install xgboost
xg = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss")
xg.fit(X_train,Y_train)

Y_predict_xg = xg.predict(X_test)

# Decision Trees
DT = DecisionTreeClassifier()
DT.fit(X_train,Y_train)

Y_predict_DT = DT.predict(X_test)

# KNN
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_train, Y_train)

Y_predict_KNN = KNN.predict(X_test)

# SVM
from sklearn.svm import SVC
SVM = SVC(kernel='rbf', C=1.0, gamma='scale')
SVM.fit(X_train,Y_train)

Y_predict_SVM = SVM.predict(X_test)

# Part 5: Model Evaluation
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# Accuracy Score
accuracy_logistic = accuracy_score(Y_test, Y_predict_logistic)
accuracy_rf = accuracy_score(Y_test, Y_predict_rf)
accuracy_ada = accuracy_score(Y_test, Y_predict_ada)
accuracy_xg = accuracy_score(Y_test, Y_predict_xg)
accuracy_DT = accuracy_score(Y_test, Y_predict_DT)
accuracy_KNN = accuracy_score(Y_test, Y_predict_KNN)
accuracy_SVM = accuracy_score(Y_test, Y_predict_SVM)

# Precision Score
precision_logistic = precision_score(Y_test, Y_predict_logistic, average = 'weighted')
precision_rf = precision_score(Y_test, Y_predict_rf, average = 'weighted')
precision_ada = precision_score(Y_test, Y_predict_ada, average = 'weighted')
precision_xg = precision_score(Y_test, Y_predict_xg, average = 'weighted')
precision_DT = precision_score(Y_test, Y_predict_DT, average = 'weighted')
precision_KNN = precision_score(Y_test, Y_predict_KNN, average = 'weighted')
precision_SVM = precision_score(Y_test, Y_predict_SVM, average = 'weighted')


# Recall Score
recall_logistic = recall_score(Y_test, Y_predict_logistic, average = 'weighted')
recall_rf = recall_score(Y_test, Y_predict_rf, average = 'weighted')
recall_ada = recall_score(Y_test, Y_predict_ada, average = 'weighted')
recall_xg = recall_score(Y_test, Y_predict_xg, average = 'weighted')
recall_DT = recall_score(Y_test, Y_predict_DT, average = 'weighted')
recall_KNN = recall_score(Y_test, Y_predict_KNN, average = 'weighted')
recall_SVM = recall_score(Y_test, Y_predict_SVM, average = 'weighted')


# F1 Score
f1_logistic = f1_score(Y_test, Y_predict_logistic, average = 'weighted')
f1_rf = f1_score(Y_test, Y_predict_rf, average = 'weighted')
f1_ada = f1_score(Y_test, Y_predict_ada, average = 'weighted')
f1_xg = f1_score(Y_test, Y_predict_xg, average = 'weighted')
f1_DT = f1_score(Y_test, Y_predict_DT, average = 'weighted')
f1_KNN = f1_score(Y_test, Y_predict_KNN, average = 'weighted')
f1_SVM = f1_score(Y_test, Y_predict_SVM, average = 'weighted')


# Use table to compare models
from tabulate import tabulate

data = [
    ['Logistic Regression',accuracy_logistic,precision_logistic, recall_logistic, f1_logistic],
    ['Random Forest',accuracy_rf,precision_rf, recall_rf, f1_rf],
    ['AdaBoost',accuracy_ada,precision_ada, recall_ada, f1_ada],
    ['XGBoost',accuracy_xg,precision_xg, recall_xg, f1_xg],
    ['Decision Tree',accuracy_DT,precision_DT, recall_DT, f1_DT],
    ['KNN',accuracy_KNN,precision_KNN, recall_KNN, f1_KNN],
    ['SVM',accuracy_SVM,precision_SVM, recall_SVM, f1_SVM]
    ]

header = ['Model Name', 'Accuracy', 'Precision', 'Recall','F1-Score']

table = tabulate(data, header, tablefmt ='fancy_grid', floatfmt=('.0%', '.2%', '.2%', '.2%', '.2%'))
print(table)
