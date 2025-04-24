import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def concat_statcast(input1, input2, output):
  try:
    df1 = pd.read_csv(input1)
    df2 = pd.read_csv(input2)
    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_df.to_csv(output, index=False)
  except:
    print("file not found error")

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
    print("file not found error")

input3 = 'Complete_Statcast.csv'
input4 = range(2015, 2025)
columns = ['WAR', 'R', 'OPS+', 'rOBA', 'Rbat+']
output = "Complete_Data.csv"
merge_data(input3, input4, output, columns)
data = pd.read_csv(output)  

key_columns = ['xba', 'barrel_batted_rate', 'player_age', 'WAR', 'k_percent', 'bb_percent', 'on_base_plus_slg', 'Rbat+']

# Univariate Analysis
print(data[key_columns].describe())

# Histograms
plt.figure(figsize=(16, 12))
for i, col in enumerate(key_columns, 1):
    plt.subplot(3, 3, i)
    sns.histplot(data[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('univariate.png')
plt.close()

# Bivariate Analysis
print(data[key_columns].corr())

# Bivariate Scatterplots
plt.figure(figsize=(12, 8))
# xBA vs WAR
plt.subplot(2, 2, 1)
sns.scatterplot(x='xba', y='WAR', data=data)
plt.title('xBA vs WAR')
# Barrel% vs OPS
plt.subplot(2, 2, 2)
sns.scatterplot(x='barrel_batted_rate', y='WAR', data=data)
plt.title('Barrel% vs WAR')
# Age vs WAR
plt.subplot(2, 2, 3)
sns.scatterplot(x='player_age', y='WAR', data=data)
plt.title('Player Age vs WAR')
# K% vs BB%
plt.subplot(2, 2, 4)
sns.scatterplot(x='Rbat+', y='WAR', data=data)
plt.title('Rbat+ vs WAR')
plt.tight_layout()
plt.savefig('bivariate.png')
plt.close()

# Multivariate Analysis
sns.pairplot(data[key_columns], diag_kind='kde', corner=True)
plt.suptitle('Pair Plot of Key Variables')
plt.savefig('multivariate.png')
plt.close()