# coding: utf-8
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

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

# Plot only median income histograms
housing_df["median_income"].hist()
plt.show()

# Create income category attribute
housing_df["income_cat"] = pd.cut(
            housing_df["median_income"],
            bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
            labels=[1, 2, 3, 4, 5]
)
housing_df["income_cat"].hist()
plt.show()


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing_df, housing_df["income_cat"]):
    strat_train_set = housing_df.loc[train_index]
    strat_test_set = housing_df.loc[test_index]
