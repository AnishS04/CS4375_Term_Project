import pandas as pd
import seaborn as sns
import numpy as np

### Read in Data ###
url = "https://raw.githubusercontent.com/AnishS04/CS4375_Term_Project/refs/heads/main/AAPL.csv"
df = pd.read_csv(url)

### Data Cleaning ### 
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')   # convert to datetime and sort chronologically

df['Open'] = pd.to_numeric(df['Open'], errors='coerce')    # convert numeric columns to floats
df['High'] = pd.to_numeric(df['High'], errors='coerce')
df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df['Adj Close'] = pd.to_numeric(df['Adj Close'], errors='coerce')
df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

df = df.sort_values('Date').reset_index(drop=True)         # sort by date 

df = df.ffill().bfill()                                    # handle missing values

df = df.drop_duplicates()                                  # remove duplicates

min_val = df['Close'].min()                                # normalize target variable (Close) to [0, 1] 
max_val = df['Close'].max()
df['Close_scaled'] = (df['Close'] - min_val) / (max_val - min_val + 1e-8)

