import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import a dataset
from sklearn import impute

dataset = pd.read_csv('data_folder/chapter2/Data.csv')
real_data = np.copy(dataset)
# This is done because i want to do it my way. And np is smart with slicing :D
X, y = real_data[:, 0:2], real_data[:, 3]
# print(X)
# print(y)

# Take care of missing data
from sklearn.preprocessing import Imputer
imputer = impute.SimpleImputer()
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# encode categorical data from text to numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X = LabelEncoder()
X[:, 0] = label_encoder_X.fit_transform(X[:, 0]) # make the column of text into encoded values such as 0, 1, 2
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y) # make the column of text into encoded values such as 0, 1, 2
# Onehot encode the countries to make sure the ML algorithm doesnt think any country thinks a country is more
# important than any other
one_hot_encoder = OneHotEncoder(categorical_features = [0])
X = one_hot_encoder.fit_transform(X).toarray()
print(X)
print(y)

# Splitting the dataset into a training set and a test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state= 0) # Test size is how much of the data to use for testing the model. 0.2 is 20%

