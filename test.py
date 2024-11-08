import pandas as pd

df = pd.read_csv(
    "/Users/eva/Desktop/Kaggle/Kaggle Time Series Course/book_sales.csv",
    index_col='Date',
    parse_dates=['Date'],
).drop('Paperback', axis=1)

df.head()
print(df)