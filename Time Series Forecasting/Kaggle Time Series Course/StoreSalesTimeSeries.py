from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data_dir = Path('/Users/eva/Desktop/Kaggle/KaggleTimeSeriesCourse')
comp_dir = Path('/Users/eva/Desktop/Kaggle/store-sales-time-series-forecasting')

ar = pd.read_csv(data_dir/'ar.csv')

dtype = {
    'store_nbr' : 'category',
    'family' : 'category',
    'sales' : 'float32',
    'onpromotion' : 'uint64',
}

store_sales = pd.read_csv(comp_dir/'train.csv',
dtype=dtype, 
parse_dates = ['date'],
infer_datetime_format = True,
)

store_sales = store_sales.set_index('date').to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family'], append=True)
# print(store_sales)

average_sales = store_sales.groupby('date').mean()['sales']
#  print(average_sales)



#Fitting a time-step feature

from sklearn.linear_model import LinearRegression

df = average_sales.to_frame()

##Creating a time-dummy

df ['Time'] = np.arange(len(average_sales.index))
# print(df.head)

##Creating training data

X = df.loc[:,['Time']]
y = df.loc[:, 'sales']

##Training the model

model = LinearRegression().fit(X,y)

y_pred = pd.Series(model.predict(X), index = X.index)
plot_params = dict(
    color = '0.75',
    style = ".-",
    markeredgecolor = '0.25',
    markerfacecolor = '0.25',
    legend = False

)

## TIme plot of total store sales
ax = y.plot(**plot_params, alpha=0.5)
ax = y_pred.plot(ax=ax, linewidth=3)
ax.set_title('Time Plot of Total Store Sales');

#Fitting a lag feature to store sales

df ['Lag_1'] = df['sales']. shift(1)

X = df.loc[:,['Lag_1']].dropna() #dataframe
y = df.loc[:,'sales'] #Series

y, X = y.align(X, join = 'inner')

##Fitting a linear model

model = LinearRegression().fit(X,y)

y_pred = pd.Series(model.predict(X), index = X.index)

fig, ax = plt.subplots()
ax.plot(X['Lag_1'],y,'.', color = '0.25')
ax.plot(X['Lag_1'], y_pred)
ax.set(aspect='equal', ylabel='sales', xlabel='Lag_1', title='Lag Plot of Average Sales');
