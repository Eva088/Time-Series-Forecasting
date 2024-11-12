from pathlib import Path
from warnings import simplefilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

simplefilter('ignore')

#Setting matplotlib defaults

plt.style.use('seaborn-v0_8-whitegrid')

plt.rc('figure',
autolayout = True,
figsize = (11,5))

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

#Loading the tunnel traffic dataset

data_dir = Path('/Users/eva/Desktop/Kaggle/KaggleTimeSeriesCourse')
tunnel = pd.read_csv(data_dir/'tunnel.csv', index_col = 'Day', parse_dates = ['Day'])
tunnel = tunnel.to_period('D')

#Creating a moving average plot

moving_average = tunnel.rolling(
    window = 365, 
    center = True,
    min_periods=183).mean()


ax = tunnel.plot(style = '.',color = '0.5')
ax = moving_average.plot(ax=ax, linewidth = 3, title = 'Tunnel Traffic - 365 Moving Average', legend = False)

#Engineering the time-dummy variable

from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression
dp = DeterministicProcess(
    index = tunnel.index, 
    order = 1, #time dummy(Trend) 
    constant = True, #bias
    drop = True #to avoid collinearity 
)


X = dp.in_sample()

y = tunnel['NumVehicles']

model = LinearRegression(fit_intercept = False).fit(X,y)

y_pred = pd.Series(model.predict(X), index = X.index)


## Time Series Plot 
ax = tunnel.plot(style=".", color="0.5", title="Tunnel Traffic - Linear Trend")
ax = y_pred.plot(ax=ax, linewidth=3, label="Trend")

X = dp.out_of_sample(steps = 30)
y_fore = pd.Series(model.predict(X), index = X.index)



ax = tunnel['2005-05':].plot(**plot_params, title = 'Tunnel Traffic - Linear Trend Forecast')
ax = y_pred['2005-05':].plot(ax = ax, linewidth = 3, label = 'Trend')
ax = y_fore.plot(ax = ax, linewidth=3, label = 'Trend Forecast', color = 'C3')
ax = ax.legend()





