# This is the code for Task 4 Oasis byte internship
# Task 4: Email Spam Detection with Machine Learning
# The Code is written by Muhammad Mudassir Majeed
# The date is Jan-24.
# Dataset: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

#-----------------------------------------------------------------------------#

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Part 1: Load Data set
import chardet
with open('Task 4\spam.csv','rb') as file:
    result = chardet.detect(file.read())
    
encoding = result['encoding']
data = pd.read_csv('Task 4\spam.csv',encoding=encoding)

# Part 2: EDA
pd.set_option('display.max_columns', None)
data.head()

# Rename Columns
data = data.rename(columns = {'v1': 'Class', 'v2': 'Message'})
data.head()

# Check Null Values
data.isnull().sum()

# Check Duplicate Values
data.duplicated().sum()

# Check class imbalances
data['Class'].value_counts()

# We have severe class imbalances

# Part 3: Pre-processing

# Drop Unnecessary columns
data = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis = 1)

# Open up Contractions
import contractions
data['Message_exp'] = data['Message'].apply(lambda x: [contractions.fix(word) for word in x.split()])

# Convert all words to lower class
data['Message_exp'] = data['Message_exp'].apply(lambda x: [word.lower() for word in x])


# Split into X and Y
X = data['Message_exp']
X = X.str.join(' ')
Y = pd.DataFrame(data['Class'])

# Encoding of Target Variable
from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
Y_enc = encode.fit_transform(Y) 
Y = pd.DataFrame(data= Y_enc, columns=Y.columns)


# Part 4: Feature Extraction for NLP using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
TFIDF = TfidfVectorizer()
X = TFIDF.fit_transform(X)

# Part 5: Model Training
# Split into train and test models
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state= 42, stratify=Y)


# Apply SMOTE to balance data for training set
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_train_rs, Y_train_rs = smote.fit_resample(X_train, Y_train)


# Decision Trees
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(X_train_rs, Y_train_rs)

Y_predict_DT = DT.predict(X_test)

# Naive Bayes
from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()
NB.fit(X_train_rs, Y_train_rs)

Y_predict_NB = NB.predict(X_test)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train_rs, Y_train_rs)

Y_predict_rf = rf.predict(X_test)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(X_train_rs, Y_train_rs)

Y_predict_lg = lg.predict(X_test)

# SVM
from sklearn.svm import SVC
SVM = SVC(kernel= 'rbf', C = 1.0, gamma = 'scale')
SVM.fit(X_train_rs, Y_train_rs)

Y_predict_SVM = SVM.predict(X_test)

# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
base_classifier = DecisionTreeClassifier(max_depth= 3)
ada = AdaBoostClassifier(base_classifier, n_estimators= 50)
ada.fit(X_train_rs, Y_train_rs)

Y_predict_ada = ada.predict(X_test)

# XG Boost
import xgboost as xgb
xg = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss")
xg.fit(X_train_rs, Y_train_rs)

Y_predict_xg = xg.predict(X_test)

# Part 6: Model Evaluation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Accuracy
accuracy_DT = accuracy_score(Y_test, Y_predict_DT)
accuracy_NB = accuracy_score(Y_test, Y_predict_NB)
accuracy_rf = accuracy_score(Y_test, Y_predict_rf)
accuracy_lg = accuracy_score(Y_test, Y_predict_lg)
accuracy_SVM = accuracy_score(Y_test, Y_predict_SVM)
accuracy_ada = accuracy_score(Y_test, Y_predict_ada)
accuracy_xg = accuracy_score(Y_test, Y_predict_xg)

# Precision
precision_DT = precision_score(Y_test, Y_predict_DT)
precision_NB = precision_score(Y_test, Y_predict_NB)
precision_rf = precision_score(Y_test, Y_predict_rf)
precision_lg = precision_score(Y_test, Y_predict_lg)
precision_SVM = precision_score(Y_test, Y_predict_SVM)
precision_ada = precision_score(Y_test, Y_predict_ada)
precision_xg = precision_score(Y_test, Y_predict_xg)

# Recall
recall_DT = recall_score(Y_test, Y_predict_DT)
recall_NB = recall_score(Y_test, Y_predict_NB)
recall_rf = recall_score(Y_test, Y_predict_rf)
recall_lg = recall_score(Y_test, Y_predict_lg)
recall_SVM = recall_score(Y_test, Y_predict_SVM)
recall_ada = recall_score(Y_test, Y_predict_ada)
recall_xg = recall_score(Y_test, Y_predict_xg)

# F1 Score
f1_DT = f1_score(Y_test, Y_predict_DT)
f1_NB = f1_score(Y_test, Y_predict_NB)
f1_rf = f1_score(Y_test, Y_predict_rf)
f1_lg = f1_score(Y_test, Y_predict_lg)
f1_SVM = f1_score(Y_test, Y_predict_SVM)
f1_ada = f1_score(Y_test, Y_predict_ada)
f1_xg = f1_score(Y_test, Y_predict_xg)

# Data Presentation
from tabulate import tabulate

table_data = [
            ['Decision Tree', accuracy_DT, precision_DT, recall_DT, f1_DT],
            ['Naive Bayes', accuracy_NB, precision_NB, recall_NB, f1_NB],    
            ['Random Forest', accuracy_rf, precision_rf, recall_rf, f1_rf],
            ['Logistic Classifier', accuracy_lg, precision_lg, recall_lg, f1_lg],
            ['Support Vector Machine', accuracy_SVM, precision_SVM, recall_SVM, f1_SVM],
            ['AdaBoost', accuracy_ada, precision_ada, recall_ada, f1_ada],
            ['XGBoost', accuracy_xg, precision_xg, recall_xg, f1_xg]
            ]

header = ['Model Name', 'Accuracy', 'Precision', 'Recall', 'F1-Score']

table = tabulate(table_data, headers= header, tablefmt='fancy_grid', 
                 floatfmt=('.3f','.3f','.3f','.3f','.3f',))

print(table)