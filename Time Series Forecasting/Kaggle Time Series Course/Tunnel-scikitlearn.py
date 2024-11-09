#Tunnel Traffic is a time series describing the number of vehicles traveling through the Baregg Tunnel in Switzerland each day from November 2003 to November 2005.

from pathlib import Path
from warnings import simplefilter

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

simplefilter('ignore')

# Setting matplotlib defaults

plt.style.use('seaborn-v0_8-darkgrid')
plt.rc(
    'figure',
    autolayout = True,
    figsize = (11, 4)
)

plt.rc(
    'axes',
    labelweight = 'bold',
    labelsize = 'large',
    titleweight = 'bold',
    titlesize = 14,
    titlepad = 10
)

plot_params = dict(
    color = '0.75',
    style = ".-",
    markeredgecolor = '0.25',
    markerfacecolor = '0.25',
    legend = False

)

#Loading the Tunnel Traffic dataset
data_dir = Path('/Users/eva/Kaggle-Projects/Time Series Forecasting/Kaggle Time Series Course')
tunnel = pd.read_csv(data_dir/'tunnel.csv', parse_dates = ['Day'], index_col = ['Day'])
tunnel = tunnel.to_period()


#Time-step feature

df = tunnel.copy()
df['Time']=np.arange(len(df.index))


#Fitting the linear regression model using scikit-learn

from sklearn.linear_model import LinearRegression

# Training Data
X = df.loc[:,['Time']] #Features
y = df.loc[:,'NumVehicles'] #target
# print(X.head())
# print(Y)

#Training the model
model = LinearRegression().fit(X,y)


#Storing the fitted values as a time series with the same time index as the training data

y_pred = pd.Series(model.predict(X), index = X.index)
print(y_pred)
print(model.coef_ )
print(model.intercept_)


#The model created is : Vehicles = 22.5 * Time + 98176. 

#Creating a time series plot

data = y.plot(**plot_params)


ax = y_pred.plot(ax = data, linewidth = 2)
ax.set_title ('Time Plot of Tunnel Traffic')
# #%%
# plt.show()



