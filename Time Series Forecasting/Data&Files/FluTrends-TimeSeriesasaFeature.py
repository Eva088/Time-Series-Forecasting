#Examining the dataset 'Flu Trends' that contains records of doctor's visits for the flu for week between 2009 and 2016. The goal is to forecast the number of flu cases for the coming weeks.

## We will first forecast doctor's visits using the lag features, and later forecast it using lags of the another set of time seroes anut the flu-related search terms as captured by Google trends. 

from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import periodogram
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_pacf 

simplefilter("ignore")

# Set Matplotlib defaults
plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 4))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
)

def lagplot(x, y=None, lag=1, standardize=False, ax=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    x_ = x.shift(lag)
    if standardize:
        x_ = (x_ - x_.mean()) / x_.std()
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y
    else:
        y_ = x
    corr = y_.corr(x_)
    if ax is None:
        fig, ax = plt.subplots()
    scatter_kws = dict(
        alpha=0.75,
        s=3,
    )
    line_kws = dict(color='C3', )
    ax = sns.regplot(x=x_,
                     y=y_,
                     scatter_kws=scatter_kws,
                     line_kws=line_kws,
                     lowess=True,
                     ax=ax,
                     **kwargs)
    at = AnchoredText(
        f"{corr:.2f}",
        prop=dict(size="large"),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)
    return ax


def plot_lags(x, y=None, lags=6, nrows=1, lagplot_kwargs={}, **kwargs):
    import math
    kwargs.setdefault('nrows', nrows)
    kwargs.setdefault('ncols', math.ceil(lags / nrows))
    kwargs.setdefault('figsize', (kwargs['ncols'] * 2, nrows * 2 + 0.5))
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
    for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):
        if k + 1 <= lags:
            ax = lagplot(x, y, lag=k + 1, ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lag {k + 1}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis('off')
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    return fig

data_dir = Path('/Users/eva/Desktop/Kaggle/KaggleTimeSeriesCourse')
flu_trends = pd.read_csv(data_dir/'flu-trends.csv')

flu_trends.set_index(
    pd.PeriodIndex(flu_trends.Week, freq = 'W'), inplace = True
)
flu_trends.drop('Week', axis = 1, inplace = True)

# ax = flu_trends.FluVisits.plot(title = 'Flu Trends', **plot_params)

# ax = ax.set(ylabel = 'Office Visits')



##The plot shows irregular cycles instead of a regular seasonality: the peak tends to occur around the new year, but sometimes earlier or later, sometimes larger or smaller. Modeling these cycles with lag features will allow the forecaster to react dynamically to changing conditions instead of being constrained to exact dates and time as with seasonal features. 


#Lag Plots and Autocorrelation Plots

# _ = plot_lags(flu_trends.FluVisits, lags = 12, nrows = 2)
# _ = plot_pacf(flu_trends.FluVisits, lags = 12)



##The lag plots indicate that the relationship of flu visits to its lags is mostly linear, while the partial autocreelations suggest the dependence can be captured using lags 1,2,3,4.

#Creating a lag time series in Pandas

def make_lags(ts, lags):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(1, lags + 1)
        },
        axis=1)

X = make_lags(flu_trends.FluVisits, lags = 4)
X = X.fillna(0.0)

y = flu_trends.FluVisits.copy()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=60, shuffle= False)

model = LinearRegression()

model.fit(X_train,y_train)

y_pred = pd.Series(model.predict(X_train), index = y_train.index)
y_fore = pd.Series(model.predict(X_test), index = y_test.index)

# ax = y_train.plot(**plot_params)
# ax = y_test.plot(**plot_params)
# ax = y_pred.plot(ax =ax)
# ax = y_fore.plot(ax = ax, color = 'C3')

#%%

plt.show()

#Incorporating the leading indicators in the training data (e.g search for phrases such as Flu Cough)

ax = flu_trends.plot(
    y = ['FluCough', 'FluVisits'],
    secondary_y = 'FluCough'

)

#%%

plt.show()



