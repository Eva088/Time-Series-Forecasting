#Source: https://www.kaggle.com/code/ryanholbrook/linear-regression-with-time-series/tutorial

import pandas as pd

#Recording the number of hardcover book sales at a retail store over 30 days 

df = pd.read_csv("../Kaggle-Projects/Time Series Forecasting/Kaggle Time Series Course/book_sales.csv",
index_col = 'Date'). drop('Paperback', axis = 1)
print(df)

#Linear Regression algorithm learns how to make a weighted sum from its input features

##Target = weight_1 * feature_1 + weight_2 * Feature_2 + bias (Orinary Least Squares - since it chooses values that minimize the squared error between the target and the predictions)

#The weights are called regression coefficients and the bias is also called an intercept. 

#There are two features unique to time-series: Time-step features and lag features. Time step features are features we can derive directly from the time index. The most basic time-step feature is the time dummy, which counts off the time steps in the series from beginning to end. 

import numpy as np
df['Time'] = np.arange(len(df.index))
print(df.head())

#Linear regression with time dummy produces the model: target = weight * feature + bias

#Creating a time plot where Time forms the x-axis




