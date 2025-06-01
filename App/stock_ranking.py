import os
import pandas as pd
import numpy as np
from collections import defaultdict

'''
This script calculates the volatility of stocks based on their trading data.
It reads multiple CSV files containing stock data, computes the weighted average price (WAP),
calculates log returns, and then computes the realized volatility for each stock over specified time buckets.
It then ranks the stocks based on their mean volatility and identifies the top and bottom 10 stocks.
'''

# List all CSV files (assuming they're named like 'stock_1.csv', 'stock_2.csv', etc.)
# Adjust the path to your directory containing the stock data files
stock_files = [f for f in os.listdir('/Users/cuongtrantrong/Documents/study/usyd/Y3/DATA3888/Group/individual_book_train') if f.endswith('.csv')] 

# Get the absolute path of the files
stock_files = [os.path.join('/Users/cuongtrantrong/Documents/study/usyd/Y3/DATA3888/Group/individual_book_train', f) for f in stock_files]

# Initialize a dictionary to hold time_id and volatility data
timeid_vol_dict = defaultdict(list)

for file in stock_files:
    df = pd.read_csv(file, on_bad_lines='skip')
    stock_id = file.split('/')[-1].split('.')[0]  # Extract stock ID from filename
    
    df['bid_price1'] = pd.to_numeric(df['bid_price1'], errors='coerce')
    df['ask_price1'] = pd.to_numeric(df['ask_price1'], errors='coerce')
    df['bid_size1'] = pd.to_numeric(df['bid_size1'], errors='coerce')
    df['ask_size1'] = pd.to_numeric(df['ask_size1'], errors='coerce')
    df['seconds_in_bucket'] = pd.to_numeric(df['seconds_in_bucket'], errors='coerce')
    
    # Compute WAP (Weighted Average Price)
    df['wap'] = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
    
    df['time_bucket'] = np.ceil(df['seconds_in_bucket'] / 20)
    df['log_ret'] = np.log(df['wap']).groupby(df['time_id']).diff().fillna(0)
    
    rv = df.groupby(['time_id', 'time_bucket'])['log_ret'].apply(lambda x: np.sqrt(np.sum(x ** 2)))
    for (time_id, bucket), vol in rv.items():
        timeid_vol_dict[time_id].append({
            'stock_id': stock_id,
            'time_bucket': bucket,
            'volatility': vol
    })
        
global_vols = pd.DataFrame([
    {'time_id': tid, 'time_bucket': x['time_bucket'], 'stock_id': x['stock_id'], 'volatility': x['volatility']}
    for tid in timeid_vol_dict for x in timeid_vol_dict[tid]
])

thresholds_q1 = global_vols.groupby('stock_id')['volatility'].quantile(0.25).to_dict()
# print(thresholds_q1)

thresholds_q3 = global_vols.groupby('stock_id')['volatility'].quantile(0.75).to_dict()
# print(thresholds_q3)

# Finding the mean votatility for all 1 stocks, and then ranking them
mean_volatility = global_vols.groupby('stock_id')['volatility'].mean().to_dict()

# Sort the stocks by mean volatility
sorted_stocks = sorted(mean_volatility.items(), key=lambda x: x[1], reverse=True)

# Get the top 10 stocks
top_10_stocks = sorted_stocks[:10]

# Get the bottom 10 stocks
bottom_10_stocks = sorted_stocks[-10:]

# Get the stock IDs
print("Top 10 stocks by mean volatility:")
for stock in top_10_stocks:
    print(f"Stock ID: {stock[0]}, Mean Volatility: {stock[1]}")
print("Bottom 10 stocks by mean volatility:")
for stock in bottom_10_stocks:
    print(f"Stock ID: {stock[0]}, Mean Volatility: {stock[1]}")

''' Since the code will take a lot of time to run, here is our output, should look like this:
Top 10 stocks by mean volatility:
Stock ID: stock_18, Mean Volatility: 0.0013716133496722517
Stock ID: stock_80, Mean Volatility: 0.0012951858690466956
Stock ID: stock_6, Mean Volatility: 0.0012326570255715616
Stock ID: stock_75, Mean Volatility: 0.001101891681449697
Stock ID: stock_27, Mean Volatility: 0.001089462514601631
Stock ID: stock_3, Mean Volatility: 0.001086316770606476
Stock ID: stock_97, Mean Volatility: 0.0010712851746516958
Stock ID: stock_37, Mean Volatility: 0.0010444838061548314
Stock ID: stock_62, Mean Volatility: 0.0009674189884560752
Stock ID: stock_40, Mean Volatility: 0.0009153240960849667
Bottom 10 stocks by mean volatility:
Stock ID: stock_93, Mean Volatility: 0.00042664520486230634
Stock ID: stock_2, Mean Volatility: 0.0004260689497661037
Stock ID: stock_64, Mean Volatility: 0.00042185364246157957
Stock ID: stock_69, Mean Volatility: 0.00042123675559908375
Stock ID: stock_47, Mean Volatility: 0.0003868635340694366
Stock ID: stock_41, Mean Volatility: 0.0003651521366867796
Stock ID: stock_125, Mean Volatility: 0.000350456134313042
Stock ID: stock_46, Mean Volatility: 0.00033704466077430945
Stock ID: stock_29, Mean Volatility: 0.00032740861690401947
Stock ID: stock_43, Mean Volatility: 0.0002588527504441149
'''
