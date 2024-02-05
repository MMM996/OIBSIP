# This is the code for Task 5 Oasis byte internship
# Task 5: Sales Prediction Using Python
# This is a Regression Prediction Problem
# The Code is written by Muhammad Mudassir Majeed
# The date is Jan-24.
# Dataset: https://www.kaggle.com/datasets/bumba5341/advertisingcsv

#-----------------------------------------------------------------------------#
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Part 1: Load Dataset
data = pd.read_csv('Task 5\Advertising.csv')

# Part 2: Preliminary EDA
data.head(50)
data.info()
data.describe()

# Check Null Values
data.isnull().sum()     # No Null Values

# Check Duplicate value
data.duplicated().sum()     # No Duplicate Values

# Part 3: Pre-processing

# Drop unnecessary columns
data = data.drop(['Unnamed: 0'], axis = 1)
data.head()
data.info()
data.describe()

# Split in X and Y
X = data.drop(['Sales'], axis =1)
Y = data['Sales']

# Standardize
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_standardized = scalar.fit_transform(X)
X_stand = pd.DataFrame(data=X_standardized, columns=X.columns)

# Check outliers using Boxplots
grid_row, grid_col = 2,2
plt.figure(figsize=(12,10))
plt.suptitle('Box Plots for all Features')

for i, column in enumerate(X_stand.columns, 1):
    plt.subplot(grid_row, grid_col,1)
    sns.boxplot(data = X_stand)
    plt.title('Box plot for feature {}'.format(column))
    
plt.tight_layout(rect=([0.00,0.03,0.99,1]))   
plt.show()

# We have outlier values in 'Newspaper' Column
column_out = 'Newspaper'
Q1 = X_stand[column_out].quantile(0.25)
Q3 = X_stand[column_out].quantile(0.75)
IQR = Q3-Q1
Upper = Q3 + IQR*(1.5)
Lower = Q1 - IQR*(1.5)

# identify index with outliers
outliers_mask = (X_stand['Newspaper'] < Lower) | (X_stand['Newspaper'] > Upper)
outliers_indices = X_stand.index[outliers_mask]
# X_stand = X_stand.query(f'{Lower} <= {column_out} <= {Upper}')

X_stand = X_stand.drop(outliers_indices)
X_stand = X_stand.reset_index(drop = True)

Y = Y.drop(outliers_indices)
Y = Y.reset_index(drop= True)

# Check Distributions
grid_row, grid_col = 2,2
plt.figure(figsize=(12,10))
plt.suptitle('Distributions for features')

for i, column in enumerate(X_stand.columns,1):
    plt.subplot(grid_row, grid_col, i)
    sns.histplot(data= X_stand[column], kde= True)
    plt.title(f'Distribution for {column}')
    
plt.tight_layout(rect=([0.00,0.03,0.99,1]))
plt.show()
 
# Observe Correlations
f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(X_stand.corr(), annot = True, linewidth = .5, ax=ax, fmt= '.1f')

# Part 4: Model Training
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_stand, Y, test_size=0.2, random_state=42)

# Linear Regression
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(X_train, Y_train)

Y_predict_lr = linear.predict(X_test)

# KNN
from sklearn.neighbors import KNeighborsRegressor
KNN = KNeighborsRegressor(n_neighbors=3)
KNN.fit(X_train, Y_train)

Y_predict_KNN = KNN.predict(X_test)

# Random Forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, Y_train)

Y_predict_rf = rf.predict(X_test)


# Part 5: Model Evaluation
from sklearn.metrics import r2_score, mean_squared_error

# r2_score
r2_lr = r2_score(Y_test, Y_predict_lr)
r2_KNN = r2_score(Y_test, Y_predict_KNN)
r2_rf = r2_score(Y_test, Y_predict_rf)

# MSE
MSE_lr = mean_squared_error(Y_test, Y_predict_lr)
MSE_KNN = mean_squared_error(Y_test, Y_predict_KNN)
MSE_rf = mean_squared_error(Y_test, Y_predict_rf)

# Present in a table
from tabulate import tabulate

data = [
        ['Linear Regression', r2_lr, MSE_lr],
        ['KNN', r2_KNN, MSE_KNN],
        ['Random Forest', r2_rf, MSE_rf]
        ]

header = ['Model Name', 'r2_score', 'Mean_squared_error']

table = tabulate(data, headers=header, tablefmt='fancy_grid',
                 floatfmt=('.3f','.3f','.3f'))
print(table)


