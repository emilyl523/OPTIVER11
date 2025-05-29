import pandas as pd

df1 = pd.read_csv('time_id_reference.csv')
df2 = pd.read_csv('order_book_feature.csv', sep='\t')
merged_df = pd.merge(df1, df2, on='time_id', how='inner')
merged_df.to_csv('order_book_feature_output.csv', index=False)

