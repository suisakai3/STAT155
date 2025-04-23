import pandas as pd

def concat_statcast(input1, input2, output):
  try:
    df1 = pd.read_csv(input1)
    df2 = pd.read_csv(input2)
    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_df.to_csv(output, index=False)
  except:
    print("file not found error")

input1 = "~/STAT155/Project 1/Data/Statcast.csv"
input2 = "~/STAT155/Project 1/Data/Statcast_2020.csv"
output = "~/STAT155/Project 1/Data/Complete_Statcast.csv"
concat_statcast(input1, input2, output)
