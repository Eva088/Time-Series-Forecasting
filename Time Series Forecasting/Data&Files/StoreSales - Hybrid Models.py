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

class BoostedHybrid():
    def __init__(self,model_1, model_2):
        self.model1 = model_1
        self.model2 = model_2
        self.y_columns = None

#Defining the fit method for boosted hybrid

    def fit(self, X_1, X_2, y):
   #fit self.model_1
        self.model1.fit(X_1,y)
        y_fit = pd.DataFrame(
        self.model1.predict(X_1), index = X_1.index, columns = y.columns
        
        )

        #compute residuals
        y_resid = y - y_fit
        y_resid = y_resid.stack().squeeze() # wide to long

        #fit self.model_2 on residuals
        self.model_2.fit(X_2,y_resid)
    
     

        # Save column names for predict method
        self.y_columns = y.columns
        # Save data for question checking
        self.y_fit = y_fit
        self.y_resid = y_resid


