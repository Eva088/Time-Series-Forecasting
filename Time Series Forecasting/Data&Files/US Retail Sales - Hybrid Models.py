#This project involves creating a linear regression + XGBoost hybrid to forecast US Retail sales in the years from 2016-19 using the sales data from 1992 to 2019, as collected by the US Census Bureau. 

from pathlib import Path
from warnings import simplefilter
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from xgboost import XGBRegressor

simplefilter('ignore')

#Setting Matplotlib defaults

plt.style.use("seaborn-v0_8-whitegrid")
plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 4),
    titlesize=18,
    titleweight='bold',
)
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

data_dir = Path('/Users/eva/Desktop/Kaggle/KaggleTimeSeriesCourse')
industries = ['BuildingMaterials', 'FoodAndBeverage'] 
retail = pd.read_csv( data_dir/'us-retail-sales.csv',
usecols = ['Month'] + industries,
parse_dates = ['Month'],
index_col = ['Month'],
).to_period('D').reindex(columns = industries)


retail = pd.concat({'Sales': retail}, names=['None', 'Industries'], axis=1) #Assigns the label 'Sales' to the entire retail dataframe and names the inner level 'industries'

#Using a linear regression model to learn the trend in each series

y = retail.copy()

##Creating trend features

dp = DeterministicProcess(
    index = y.index,
    order = 2,
    constant = True,
    drop = True)

X = dp.in_sample()

#Testing on the years 2016-2019.Splitting the date index instead of the dataframe directly. 

idx_train, idx_test = train_test_split(
    y.index, test_size = 12 * 4, shuffle = False
)

X_train, X_test = X.loc[idx_train, :], X.loc[idx_test, :]

y_train, y_test = y.loc[idx_train, :], y.loc[idx_test, :]

#Fitting the trend model

model = LinearRegression(fit_intercept = False)
model.fit(X_train,y_train)

#Making the prediction

y_fit = pd.DataFrame(model.predict(X_train), index = y_train.index, columns = y_train.columns)

y_pred = pd.DataFrame(model.predict(X_test), index = y_test.index, columns = y_test.columns)

axs = y_train.plot(color = '0.25', subplots = True, sharex = True)
axs = y_test.plot(color = '0.25', subplots = True, sharex = True, ax = axs)
axs = y_fit.plot(color = 'C0', subplots = True, sharex = True, ax = axs)
axs = y_pred.plot(color = 'C3', subplots = True, sharex = True, ax = axs)

for ax in axs: ax.legend([])
_ = plt.suptitle("Trends")


#Converting the series from a wide ( one time series per column ) to a long format (series indexed by categories along rows)
 
X = retail.stack()

y = X.pop('Sales')

X = X.reset_index('Industries')

#Label encoding for 'Industries' feature

for colname in X.select_dtypes(['object','category']):
    X[colname], _ = X[colname].factorize() #encoded values, unique values 
    
#Label encoding for annual seasonality

X['Month'] = X.index.month

#Creating splits

X_train, X_test = X.loc[idx_train,:], X.loc[idx_test,:]
y_train, y_test = y.loc[idx_train,:],y.loc[idx_test,:]


#Converting the trend predictions into a long format and then subtracting them from the original series to create a detrended(residuals) series that XGBoost can learn. 

y_fit = y_fit.stack().squeeze()
y_pred = y_pred.stack().squeeze()


#Creating residuals (the collection of detrended series) from the training set

y_resid = y_train - y_fit

#Training XGBoost on the residuals

xgb = XGBRegressor()
xgb.fit(X_train, y_resid)

#Adding the predicted residuals onto the predicted trends

y_fit_boosted = xgb.predict(X_train) + y_fit
y_pred_boosted = xgb.predict(X_test) + y_pred

axs = y_train.unstack(['Industries']).plot(
    color='0.25', figsize=(11, 5), subplots=True, sharex=True,
    title=['BuildingMaterials', 'FoodAndBeverage'],
)

axs = y_test.unstack(['Industries']).plot(
    color='0.25', subplots=True, sharex=True, ax=axs,
)

axs = y_fit_boosted.unstack(['Industries']).plot(
    color='C0', subplots=True, sharex=True, ax=axs,
)

axs = y_pred_boosted.unstack(['Industries']).plot(
    color='C3', subplots=True, sharex=True, ax=axs,
)

for ax in axs: ax.legend([])

#%%
plt.show()

#We can see that the trend learned by XGBoost is only as good as the trend learned by the linear regression -- in particular, XGBoost wasn't able to compensate for the poorly fit trend in the 'BuildingMaterials' series.