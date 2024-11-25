import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import DeterministicProcess
from xgboost import XGBRegressor
from pathlib import Path
from warnings import simplefilter

simplefilter('ignore')
comp_dir = Path('/Users/eva/Desktop/Kaggle/store-sales-time-series-forecasting')

store_sales = pd.read_csv(comp_dir/'train.csv',
usecols=['store_nbr','family', 'date', 'sales','onpromotion'],
dtype={'store_nbr':'category',
       'family':'category',
       'sales':'float32'},
parse_dates = ['date']
)

store_sales['date']= store_sales.date.dt.to_period('D')
store_sales = store_sales.set_index(['store_nbr','family','date']).sort_index()

family_sales = (
    store_sales
    .groupby(['family','date'])
    .mean()
    .unstack('family')
    .loc['2017']
)


#Creating a boosted hybrid by implementing the class BoostedHybrid

class BoostedHybrid(self, model_1, model_2):
    self.model1 = model_1
    self.model2 = model_2
    self.y_columns = None
    
    
