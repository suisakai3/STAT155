import pandas as pd
import os

def concat_statcast(input1, input2, output):
  try:
    df1 = pd.read_csv(input1)
    df2 = pd.read_csv(input2)
    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_df.to_csv(output, index=False)
  except:
    print("file not found error 1")

input1 = "Statcast.csv"
input2 = "Statcast_2020.csv"
output = "Complete_Statcast.csv"
concat_statcast(input1, input2, output)

def merge_data(input3, input4, output, columns):
  try:
    df1 = pd.read_csv(input3)
    df1['Player'] = df1['last_name, first_name'].str.split(', ').str[::-1].str.join(' ')
    all_years_df = []
    for i in input4:
      filename = f'{i}.csv'
      df2 = pd.read_csv(filename, skiprows=4, skipfooter=3, engine='python')
      df2['year'] = i 
      df2['Player'] = df2['Player'].str.replace(r'[*#]', '', regex=True).str.strip()
      df2_subset = df2[['Player', 'year'] + columns]
      all_years_df.append(df2_subset)
    combined_df2 = pd.concat(all_years_df, ignore_index=True)
    merged_df = pd.merge(df1, combined_df2, on=['Player', 'year'], how='inner')
    merged_df.to_csv(output, index=False)
    print(merged_df.info)
  except FileNotFoundError:
    print("file not found error 2")

input3 = 'Complete_Statcast.csv'
input4 = range(2015, 2025)
columns = ['WAR', 'R', 'OPS+', 'rOBA', 'Rbat+']
output = "Complete_Data.csv"
merge_data(input3, input4, output, columns)
  