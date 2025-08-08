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

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  

def tanh(x):
    return np.tanh(np.clip(x, -500, 500)) 

def softmax(x):
    exp_x = np.exp(x - np.max(x))  
    return exp_x / np.sum(exp_x)

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh_derivative(x):
    return 1 - x * x

def xavier_init(size_in, size_out):
    return np.random.randn(size_in, size_out) / np.sqrt(size_in)

class StockLSTM:
    def __init__(self, input_size, hidden_size, output_size, num_epochs=100, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # Initialize weights and biases for all gates
        # Forget gate
        self.wf = xavier_init(input_size + hidden_size, hidden_size)
        self.bf = np.zeros((1, hidden_size))
        
        # Input gate
        self.wi = xavier_init(input_size + hidden_size, hidden_size)
        self.bi = np.zeros((1, hidden_size))
        
        # Candidate gate
        self.wc = xavier_init(input_size + hidden_size, hidden_size)
        self.bc = np.zeros((1, hidden_size))
        
        # Output gate
        self.wo = xavier_init(input_size + hidden_size, hidden_size)
        self.bo = np.zeros((1, hidden_size))
        
        # Final output layer
        self.wy = xavier_init(hidden_size, output_size)
        self.by = np.zeros((1, output_size))
        
    
    ### CODE FORWARD PASS FUNCTION ###
    def forward(self, inputs):
        outputs = []
        ## LOGIC ##
        return outputs 

    ### CODE BACKWARD PROPOGATION FUNCTION ###
    


