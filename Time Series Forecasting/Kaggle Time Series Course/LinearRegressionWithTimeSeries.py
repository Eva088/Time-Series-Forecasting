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

#Modeling Time Dependence

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-whitegrid') #styling matplotlib graphs with seaborn

#set the default parameters for Matplotlib; customise the appearance of the plots globally, without having to modify each plot individually
plt.rc(
    'figure',
    autolayout = True, 
    figsize = (11,4),
    titlesize = 18,
    titleweight = 'bold'
)

plt.rc(
    'axes',
    labelweight = 'bold',
    labelsize = 'large', 
    titleweight = 'bold',
    titlesize = 16, 
    titlepad = 10
)

fig,ax = plt.subplots()
ax.plot('Time', 'Hardcover', data = df, color = '0.75') #creates a lineplot
ax = sns.regplot(x = 'Time', y = 'Hardcover', data = df, ci = None, scatter_kws = dict(color = '0.25')) #creates a scatterplot with a regression line 
ax.set_title('Time Plot of Hardcover Sales')






