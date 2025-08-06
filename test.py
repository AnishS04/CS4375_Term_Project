import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

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

df.set_index('Date', inplace = True)                       # Set Date as index 


### Feature Engineering ###

# Add time-based features
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['quarter'] = df.index.quarter
df['year'] = df.index.year

# Add lag features (1 to 5 days)
for lag in range(1, 6):
    df[f'Close_lag_{lag}'] = df['Close_scaled'].shift(lag)

# Add rolling statistics
df['rolling_mean_5'] = df['Close_scaled'].rolling(window=5).mean()
df['rolling_std_5'] = df['Close_scaled'].rolling(window=5).std()

df.dropna(inplace=True)                                   # Drop NaN rows from lag/rolling features 

### EDA ###

# Correlation matrix + heatmap 
plt.figure(figsize=(12, 10))                             
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Price over time plot
plt.figure(figsize=(14, 5))                              
sns.lineplot(data=df, x=df.index, y='Close_scaled')
plt.title("Scaled Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Close Scaled")
plt.grid(True)
plt.show()

# Closed_scale with 5 day Rolling Mean 
plt.figure(figsize=(14, 5))                             
sns.lineplot(data=df, x=df.index, y='Close_scaled', label='Close_scaled')
sns.lineplot(data=df, x=df.index, y='rolling_mean_5', label='5-Day Rolling Mean')
plt.title("Close Scaled vs. 5-Day Rolling Mean")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

#Daily Volatility (rolling std deviation)
plt.figure(figsize=(14, 4))
sns.lineplot(data=df, x=df.index, y='rolling_std_5', color='red')
plt.title("5-Day Rolling Standard Deviation (Volatility)")
plt.xlabel("Date")
plt.ylabel("Rolling Std Dev")
plt.grid(True)
plt.show()


### Train - Test Split ###
sequence_length = 20
split_fraction = 0.8

features = df.drop(columns=['Close', 'Adj Close'])        # Keep only features as these are redundant 

feature_data = features.values                            # Convert to numpy array
# Now a 2D Array of [n_timesteps, n_features]

#Split by time 
split_index = int(len(feature_data) * split_fraction)
train_data = feature_data[:split_index]
test_data = feature_data[split_index:]



