from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from warnings import simplefilter

simplefilter('ignore')
comp_dir = Path('/Users/eva/Desktop/Kaggle/store-sales-time-series-forecasting')

# Set Matplotlib defaults
plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
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
    legend=False,
)
def seasonal_plot(X, y, period, freq, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
    return ax


def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("365D") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax

store_sales = pd.read_csv(
    comp_dir / 'train.csv',
    usecols=['store_nbr', 'family', 'date', 'sales'],
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
    },
    parse_dates=['date'],
)

store_sales['date'] = store_sales.date.dt.to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()

average_sales = (
    store_sales
    .groupby('date').mean()
    .squeeze()
    .loc['2017']
)

# #Examining the seasonal plot

# X = average_sales.to_frame()
# X["week"] = X.index.week
# X['day'] = X.index.dayofweek
# seasonal_plot(X, y = 'sales', period = 'week', freq = 'day')

#Examining the plot periodogram - examining seasonal variance

# plot_periodogram(average_sales)

##Both the seasonal plot and periodogram show a strong weekly seasonality. The periodogram also indicates slight monthly and biweekly seasonality. It could be due to the fact that the wages are paid out biweekly, on the 15th and last day of the month.

#Creating seasonal features

y = average_sales.copy()

fourier = CalendarFourier('M', 4)

dp = DeterministicProcess(
    index = y.index,
    constant = True,
    order = 1,
    seasonal = True,
    additional_terms = [fourier],
    drop = True
)


X = dp.in_sample()

# Fitting the seasonal model

model = LinearRegression().fit(X,y)
y_pred = pd.Series(model.predict(X), index = X.index, name = 'Fitted')

# ax = y.plot(**plot_params, alpha = 0.5, title = 'Average Sales', ylabel = 'items sold')
# ax = y_pred.plot(ax = ax, label = 'Seasonal')
# ax.legend()

# #%%
# plt.show()

#Detrending/Deseasonalizing

##Removing from a series its trend or seasons is called detrending or deseasonalizing the series. 

## periodogram of the deseasonalized series

y_deseasoned = y - y_pred

# fig, (ax1,ax2) = plt.subplots (2,1, sharex= True, sharey = True, figsize = (10,7))
# ax1 = plot_periodogram(y, ax = ax1)
# ax1.set_title ('Product Sales Frequency Components')
# ax2 = plot_periodogram(y_deseasoned, ax=ax2)
# ax2.set_title ('Deseasonalized')


## The plots indicate that our model captures seasonality better compared to the original series. 

# Reading the file

data_dir = Path('/Users/eva/Desktop/Kaggle/store-sales-time-series-forecasting')
holidays = pd.read_csv(data_dir/'holidays_events.csv', parse_dates = ['date'])
holidays.columns = holidays.columns.str.strip()
holidays['description'] = holidays['description'].astype('category')
holidays['date'] = pd.to_datetime(holidays['date']).dt.to_period('D')





## National and regional holidays in the training set

holidays = (holidays
            .set_index('date')
            .query("locale in ['National', 'Regional']")
            .loc['2017' : '2017-08-15', ['description']]
            .assign(description = lambda x: x.description.cat.remove_unused_categories())
            )

print(holidays.head(10))

print(y.index)
print(y_deseasoned.index)
print(holidays.index)

#Check if all holidays dates exist in y_deseasoned index

missing_dates = holidays.index[~holidays.index.isin(y_deseasoned.index)]

if len(missing_dates) > 0:
    print("Missing dates in y_deseasoned:", missing_dates)
else:
    print("All dates in holidays are found in y_deseasoned")

# If missing dates exist, reindex y_deseasoned to match holidays
y_deseasoned_aligned = y_deseasoned.reindex(holidays.index)

#The code is essentially plotting deseasonalized sales values over time, where the x-axis represents the dates (from holidays.index) and the y-axis represents the corresponding sales values from y_deseason.

ax = y_deseasoned.plot(**plot_params)
plt.plot_date(holidays.index, y_deseasoned[holidays.index], color='C3')
ax.set_title('National and Regional Holidays');


#%%

plt.show()
