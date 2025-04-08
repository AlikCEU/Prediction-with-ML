import pandas as pd

df = pd.read_csv("3rd Assignment/bisnode_firms_fastgrowth.csv")
print("Shape of dataset:", df.shape)
print("Columns:", df.columns.tolist())
print("Target distribution:\n", df["fast_growth"].value_counts())
