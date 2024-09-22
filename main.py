# coding: utf-8
import os
import pandas as pd

# Data path
HOUSING_PATH = os.path.join("data", "housing.csv")

# Load housing data into a pandas dataframe and view the top 5 rows
housing_df = pd.read_csv(HOUSING_PATH)
housing_df.head()

# Show quick descripton of the data
housing_df.info()
