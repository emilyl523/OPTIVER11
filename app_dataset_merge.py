import pandas as pd

df_time_ref = pd.read_csv('time_id_reference.csv')  # contains 'time_id' and 'date'

df_first_half = pd.read_csv('order_book_feature.csv', sep='\t')   # 0–1799s
df_second_half = pd.read_csv('order_book_target.csv', sep='\t')   # 1800–3599s


df_first_half = df_first_half[df_first_half['time_id'].isin(df_time_ref['time_id'])]
df_second_half = df_second_half[df_second_half['time_id'].isin(df_time_ref['time_id'])]

full_hour_df = pd.concat([df_first_half, df_second_half], axis=0)
full_hour_df = pd.merge(full_hour_df, df_time_ref, on='time_id', how='left')

full_hour_df = full_hour_df.sort_values(by=['date', 'time_id', 'seconds'], ignore_index=True)
full_hour_df.to_csv('full_hour_combined.csv', index=False)
