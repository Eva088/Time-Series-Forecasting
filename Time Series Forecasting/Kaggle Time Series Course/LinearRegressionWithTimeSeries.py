import pandas as pd
df = pd.read_csv("../Kaggle-Projects/Time Series Forecasting/Kaggle Time Series Course/book_sales.csv",
index_col = 'Date'). drop('Paperback', axis = 1)

print(df)
