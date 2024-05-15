import pandas as pd

# Load CSV file into a Pandas DataFrame
df = pd.read_csv('C:\\Users\\CHITTURI SRINIVAS\\PycharmProjects\\pythonintern\\01.Data Cleaning and Preprocessing.csv')
# Determine the type of object df
print(type(df))
# Get the shape of the DataFrame
shape = df.shape

# Print the shape
print("Number of rows:", shape[0])
print("Number of columns:", shape[1])
# Summary statistics for numerical columns

print("\nSummary statistics for numerical columns:")
print(df.describe())
# Remove duplicate rows from the DataFrame without modifying the original DataFrame
df.drop_duplicates()
# Check for missing values in the DataFrame
missing_values = df.isnull()

# Print the DataFrame of missing values
print(missing_values)
# Handling missing values
# Check for missing values
print("\nMissing values in the DataFrame:")
print(df.isnull().sum())
# Get the total count of missing values in the DataFrame
total_missing_values = df.isnull().sum().sum()

# Print the total count of missing values
print("Total count of missing values:", total_missing_values)
# Fill missing values with a specific value
df_filled1 = df.fillna(0)  # Replace missing values with 0
print("\nDataFrame after filling missing values:")
print(df_filled1)
# Get the total count of missing values in the DataFrame
total_missing_values = df_filled1.isnull().sum().sum()

# Print the total count of missing values
print("Total count of missing values:", total_missing_values)
# Fill missing values using the 'pad' method (forward fill)
df_filled2 = df.fillna(method='pad')

# Print the DataFrame after filling missing values
print(df_filled2)
# Get the total count of missing values in the DataFrame
total_missing_values = df_filled2.isnull().sum().sum()

# Print the total count of missing values
print("Total count of missing values:", total_missing_values)
# Fill missing values using the 'bfill' method (backward fill)
df_filled3 = df.fillna(method='bfill')

# Print the DataFrame after filling missing values
print(df_filled3)
# Get the total count of missing values in the DataFrame
total_missing_values = df_filled3.isnull().sum().sum()

# Print the total count of missing values
print("Total count of missing values:", total_missing_values)
# Drop rows with missing values
df_cleaned = df.dropna()  # Drop rows with any missing values
print("\nDataFrame after dropping rows with missing values:")
print(df_cleaned)
# Get the total count of missing values in the DataFrame
total_missing_values = df_cleaned.isnull().sum().sum()

# Print the total count of missing values
print("Total count of missing values:", total_missing_values)
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
print(df_filled1.columns)
# Drop the 'Observation' column from the DataFrame
df_filled1.drop(['Observation'], axis=1, inplace=True)

# Check the remaining columns
print(df_filled1.columns)
Q1 = df_filled1.quantile(0.25)
Q3 = df_filled1.quantile(0.75)

# Calculate the interquartile range (IQR) for each numerical column
IQR = Q3 - Q1

# Print the interquartile range for each numerical column
print(IQR)
# Filter out outliers based on the interquartile range (IQR)
df_filled1 = df_filled1[~((df_filled1 < (Q1 - 1.5 * IQR)) | (df_filled1 > (Q3 + 1.5 * IQR))).any(axis=1)]
print(df_filled1)
# Summary statistics for numerical columns

print("\nSummary statistics for numerical columns:")
print(df.describe())