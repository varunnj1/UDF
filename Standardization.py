import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from Console.package import Functions

f = Functions()

# Get DataFrame and columns
df = f.get_args("df")
columns = f.get_args("columns")

# Extract individual column names from the string
column_names = [col.strip() for col in columns.split(",")]

# Ensure the specified columns are numeric
df[column_names] = df[column_names].apply(pd.to_numeric, errors='coerce')

# Initialize subplots for each column
fig, axs = plt.subplots(nrows=len(column_names), ncols=2, figsize=(12, 6 * len(column_names)))
fig.suptitle('Data Distribution Before and After Standardization')

# Standardize each specified column and plot histograms
for i, col in enumerate(column_names):
    # Plot histogram before standardization
    plt.subplot(len(column_names), 2, i * 2 + 1)
    plt.hist(df[col].dropna(), bins=20, color='blue', alpha=0.7)
    plt.title(f'{col} - Before Standardization')
    plt.xlabel(col)
    plt.ylabel('Frequency')

    # Standardize the column
    scaler = StandardScaler()
    standardized_col = scaler.fit_transform(df[col].values.reshape(-1, 1))

    # Plot histogram after standardization
    plt.subplot(len(column_names), 2, i * 2 + 2)
    plt.hist(standardized_col, bins=20, color='green', alpha=0.7)
    plt.title(f'{col} - After Standardization')
    plt.xlabel(f'{col}_standardized')
    plt.ylabel('Frequency')

    # Update the DataFrame with standardized data
    df[f'{col}_standardized'] = standardized_col.flatten()

    # Display standardized data
    standardized_data = pd.DataFrame({f'{col}_standardized': scaler.fit_transform(df[col].values.reshape(-1, 1)).flatten()})

    f.save_table(standardized_data)
    f.save_table(df)

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Show the plots

f.save_image(plt)
