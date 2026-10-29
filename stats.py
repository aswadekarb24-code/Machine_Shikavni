import pandas as pd
df = pd.read_csv("results/noise_results_all_models_robust_20261029_224413.csv")
print(sum(df['accuracy'])/len(df))
print(sum(df.loc[df['Dataset']=='GTSRB']['accuracy'])/len(df.loc[df['Dataset']=='GTSRB']))
print(sum(df.loc[df['Dataset']=='CTSD']['accuracy'])/len(df.loc[df['Dataset']=='CTSD']))
print(sum(df.loc[df['Dataset']=='BTSD']['accuracy'])/len(df.loc[df['Dataset']=='BTSD']))
print(df.loc[df['accuracy']== min(df['accuracy'])])
print(df.loc[df['accuracy']== max(df['accuracy'])])

import statistics
print(df.loc[df['accuracy']== statistics.median(df['accuracy'])])
