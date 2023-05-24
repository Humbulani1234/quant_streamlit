# ================================
# Data download and Data cleaning
# ================================

import pandas as pd
from datetime import datetime

options_df = pd.read_csv("options_prices.csv", header=0, sep=";")

for i in range(options_df.shape[0]):
    for j in range(2):
        
        y = options_df.iloc[i,j].replace("/", "-")
        options_df.replace(options_df.iloc[i,j], y, inplace=True)

for i in range(options_df.shape[0]):
    for j in range(2):
        
        z = datetime.fromisoformat(str(options_df.iloc[i,j]))
        options_df.replace(options_df.iloc[i,j], z, inplace=True)

options_df