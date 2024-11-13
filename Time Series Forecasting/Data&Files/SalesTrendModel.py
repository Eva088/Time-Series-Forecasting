from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import DeterministicProcess

comp_dir = Path('/Users/eva/Desktop/Kaggle/store-sales-time-series-forecasting')

dtype = {
    'store_nbr': 'category',
    'family': 'category',
    'sales': 'float32',
    'onpromotion': 'uint64',
}

plot_params = dict(
    color = '0.75',
    style = ".-",
    markeredgecolor = '0.25',
    markerfacecolor = '0.25',
    legend = False
)

store_sales = pd.read_csv(comp_dir/'train.csv',
parse_dates = ['date'],
dtype = dtype)

store_sales = store_sales.set_index('date').to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family'], append = True)
average_sales = store_sales. groupby('date').mean()['sales']

## Determining trend with a moving average plot

trend = average_sales.rolling(
    window = 365,
    min_periods = 183,
    center = True
). mean()


#Identifying the trend

# ax = average_sales.plot(**plot_params)
# ax = trend.plot(ax = ax, linewidth = 3)
# ax.set(title = 'Trend in Average Store Sales')

## Creating a trend feature using Deterministic Process

from statsmodels.tsa.deterministic import DeterministicProcess

y = average_sales.copy()  # the target

dp = DeterministicProcess(
    index = average_sales.index, 
    order = 3
)

# Create the feature set for the dates given in y.index
X = dp.in_sample()

# Create features for a 90-day forecast.
X_fore = dp.out_of_sample(steps= 90)

model = LinearRegression()

model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

ax = y.plot(**plot_params, alpha=0.5, title="Average Sales", ylabel="items sold")
ax = y_pred.plot(ax=ax, linewidth=3, label="Trend", color='C0')
ax = y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color='C3')
ax.legend();

#%%
plt.show()




