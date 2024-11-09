import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Created a dataframe with a time dummy column 
df = pd.read_csv('/Users/eva/Kaggle-Projects/Time Series Forecasting/Kaggle Time Series Course/Tunnel.csv', 
index_col= 'Day')

df['Time'] = np.arange(len(df.index))

print(df)

#Creating a time plot showing trend in the Number of Vehicles over time

plt.style.use('seaborn-v0_8-darkgrid')
plt.rc(
    'figure',
    autolayout = True, 
    figsize = (10, 8), 
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

fig, ax = plt.subplots()
# ax.plot('Time', 'NumVehicles', data = df, color = '0.75')
ax = sns.regplot( x = 'Time', y = 'NumVehicles', data = df, ci = None, scatter_kws = dict(color = '0.25'), line_kws = dict(color = 'r'))
ax.set_title('Time Plot of Number of Vehicles')

#Lag Plot showing correlation of number of vehicles compared to the previous day

df ['lag_1'] = df ['NumVehicles'].shift(1)
df = df.reindex(columns = ['NumVehicles', 'lag_1'])
print(df)

fig, ax = plt.subplots()
ax = sns.regplot(x = 'lag_1', y = 'NumVehicles', data = df, scatter_kws = dict(color = '0.25'), line_kws = dict(color = 'r'), ci = None)
ax.set_aspect(1)
ax.set_title('Lag Plot of Number of Vehicles')

#%%
plt.show()