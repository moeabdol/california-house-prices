# coding: utf-8
import os
import pandas as pd
import matplotlib.pyplot as plt

# Data path
HOUSING_PATH = os.path.join("data", "housing.csv")

# Load housing data into a pandas dataframe and view the top 5 rows
housing_df = pd.read_csv(HOUSING_PATH)
housing_df.head()

# Show quick descripton of the data
housing_df.info()

# Show unique values of the categorical field
housing_df["ocean_proximity"].value_counts()

# Show summary of all numercial values
housing_df.describe()

# Draw histograms of all numerical attributes
housing_df.hist(bins=50, figsize=(20, 15))
plt.show()
