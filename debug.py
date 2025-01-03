import pandas as pd

df = pd.read_csv("assets/reviews_for_train.csv")

print(df.isnull().sum())