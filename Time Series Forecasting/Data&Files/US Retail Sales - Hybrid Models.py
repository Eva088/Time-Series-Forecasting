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
print(retail.head())