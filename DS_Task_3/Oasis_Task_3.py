# This is the code for Task 3 Oasis byte internship
# Task 3: Car Price Prediction with Machine Learning
# The Code is written by Muhammad Mudassir Majeed
# The date is Jan-24.
# Dataset: https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars

#-----------------------------------------------------------------------------#

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Part 1: Load Dataset
data = pd.read_csv('Task 3\car data.csv')

# Part 2: Preliminary EDA
data.head()
data.info()
data.describe()

# Check for Null values
data.isnull().sum()     # No Null Values

# Check for duplicate values
data.duplicated().sum()     # 2 Duplicate values

# Check for unique values for all categorical variables
data['Car_Name'].value_counts().nunique()   # 14 Ordinal Values
data['Fuel_Type'].value_counts().nunique()  # 3 Ordinal Values
data['Selling_type'].value_counts().nunique()   # 2 Ordinal Values
data['Transmission'].value_counts().nunique()   # 2 Ordinal Values

# Check non-Zero values for column
data['Owner'].value_counts()    # Majority of column is empty
data.head(10)

# Observe Distributions
grid_row, grid_col = 3,3
plt.figure(figsize=(12,10))
plt.suptitle('Histograms for Features')

for i, column in enumerate(data.columns,1):
    plt.subplot(grid_row, grid_col,i)
    sns.histplot(data = data[column], kde=True)
    plt.title('Histogram for {}'.format(column))
    
plt.tight_layout(rect=[0,0.03,1,0.99])
plt.show()

# Part 3: Pre-processing

# Remove Duplicates
data = data.drop_duplicates()
data = data.reset_index(drop = True)
data.duplicated().sum()

# Split data into X and Y. I am trying to predict car selling price
X = data.drop(['Selling_Price'], axis =1)
Y = data['Selling_Price']

# Convert categorical values to Numeric using appropirate Encodings
X.info()
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
X['Selling_type'] = label.fit_transform(X['Selling_type'])
X['Fuel_Type'] = label.fit_transform(X['Fuel_Type'])
X['Transmission'] = label.fit_transform(X['Transmission'])

X.head(10)
X.info()

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X[['Car_Name']]).toarray()
X_encoded_df = pd.DataFrame(X_encoded, columns = encoder.get_feature_names_out(['Car_Name']))
X = pd.concat([X, X_encoded_df], axis =1 )
X = X.drop(['Car_Name'], axis =1 )

# Standardize
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_standardized = scalar.fit_transform(X)
X_stand = pd.DataFrame(data= X_standardized, columns= X.columns)
X_stand.describe()

# Observe Correlation Plots
f, ax = plt.subplots(figsize= (10,10))
sns.heatmap(X_stand.corr(), annot=True, linewidth = .5, fmt='.1f', ax=ax)

# Part 4: Model Training
# Split into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train,Y_test = train_test_split(X_stand, Y, test_size= 0.2, random_state=42)

# Linear Regression
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(X_train, Y_train)

Y_predict_linear = linear.predict(X_test)

# Random Forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=50)
rf.fit(X_train, Y_train)

Y_predict_rf = rf.predict(X_test)

# K-NN
from sklearn.neighbors import KNeighborsRegressor
KNN = KNeighborsRegressor(n_neighbors=3)
KNN.fit(X_train,Y_train)

Y_predict_KNN = KNN.predict(X_test)


# Part 5: Model Evaluation
from sklearn.metrics import r2_score, mean_squared_error

# Random Forest
r2_score_rf = r2_score(Y_test, Y_predict_rf)
mean_squared_error_rf = mean_squared_error(Y_test, Y_predict_rf)

# Linear Regression
r2_score_linear = r2_score(Y_test, Y_predict_linear)
mean_squared_error_linear = mean_squared_error(Y_test, Y_predict_linear)

# KNN
r2_score_KNN = r2_score(Y_test, Y_predict_KNN)
mean_squared_error_KNN = mean_squared_error(Y_test, Y_predict_KNN)

# Present Data in a table for different models
from tabulate import tabulate

data = [
        ['Random Forest', r2_score_rf, mean_squared_error_rf],
        ['Linear Regression',r2_score_linear, mean_squared_error_linear],
        ['KNN',r2_score_KNN, mean_squared_error_KNN]
        ]

header = ['Model Name', 'r2_score', 'Mean_Squared_Error']
table = tabulate(data, header, tablefmt ='fancy_grid', floatfmt=('.2f', '.2f', '.2f'))
print(table)