# coding: utf-8
import os
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer

# Function to save figures
def save_fig(fig_id, tight_layout=True, fig_extension="webp", resolution=300):
    path = os.path.join("images", fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

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
save_fig("histograms")

# Plot only median income histograms
housing_df["median_income"].hist()
save_fig("median_income_histogram")

# Create income category attribute
housing_df["income_cat"] = pd.cut(
            housing_df["median_income"],
            bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
            labels=[1, 2, 3, 4, 5]
)
housing_df["income_cat"].hist()
save_fig("income_cat_histograms")

# Perform stratified sampling
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing_df, housing_df["income_cat"]):
    strat_train_set = housing_df.loc[train_index]
    strat_test_set = housing_df.loc[test_index]

# Remove income_cat
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# Create copy of the training set
housing_df = strat_train_set.copy()

# Plot scatterplot
housing_df.plot(kind="scatter", x="longitude", y="latitude")
save_fig("scatterplot_1")

# Plot scatterplot with alpha
housing_df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
save_fig("scatterplot_2")

# Plot scatterplot with colorbar and legends
housing_df.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    alpha=0.4,
    s=housing_df["population"]/100,     # Marker size
    c="median_house_value",             # Marker color
    label="population",
    cmap=plt.get_cmap("jet"),           # Colormap
    colorbar=True,
    figsize=(10, 7),
)
plt.legend()
save_fig("scatterplot_3")

# Plot scatterplot with california as background image
california_img = img.imread(os.path.join("images", "california.png"))
housing_df.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    alpha=0.4,
    s=housing_df["population"]/100,     # Marker size
    c="median_house_value",             # Marker color
    label="Population",
    cmap=plt.get_cmap("jet"),           # Colormap
    colorbar=False,
    figsize=(10, 7),
)
plt.imshow(
    california_img,
    extent=[-124.55, -113.80, 32.45, 42.05],
    alpha=0.5,
    cmap=plt.get_cmap("jet"),
)
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

prices = housing_df["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar(ticks=tick_values/prices.max())
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
save_fig("scatterplot_4")

# Calculate correlations
corr_matrix = housing_df.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)

# Calculate correlations using pandas scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing_df[attributes], figsize=(12, 8))
save_fig("correlation_scatter_matrix")

# Zoom in correlation between median_house_value and median_income
housing_df.plot(kind="scatter", x="median_income", y="median_house_value",
                alpha=0.1)
save_fig("income_vs_house_value_correlation")

# Attribute combinations
housing_df["rooms_per_household"] = housing_df["total_rooms"] / housing_df["households"]
housing_df["bedrooms_per_room"] = housing_df["total_bedrooms"] / housing_df["total_rooms"]
housing_df["population_per_household"] = housing_df["population"] / housing_df["households"]

corr_matrix = housing_df.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)

# Prepare a clean training set and separate labels
housing_df = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Data cleaning (manual)
# option 1
#  housing_df.dropna(subset=["total_bedrooms"])

# option 2
#  housing_df.drop("total_bedrooms", axis=1)

# option 3
#  median = housing_df["total_bedrooms"].median()
#  housing_df["total_bedrooms"] = housing_df["total_bedrooms"].fillna(median)

# Data cleaning (SimpleImputer)
imputer = SimpleImputer(strategy="median")
housing_num = housing_df.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_df = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
