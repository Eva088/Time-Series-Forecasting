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

