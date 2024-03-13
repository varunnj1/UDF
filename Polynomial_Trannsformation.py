import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
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
fig, axs = plt.subplots(nrows=len(column_names), ncols=3, figsize=(18, 6 * len(column_names)))
fig.suptitle('Data Distribution Before and After Polynomial Feature Transformation')

# Polynomial feature transformation and plot histograms
for i, col in enumerate(column_names):
    # Plot histogram before transformation
    plt.subplot(len(column_names), 3, i * 3 + 1)
    plt.hist(df[col].dropna(), bins=20, color='blue', alpha=0.7)
    plt.title(f'{col} - Before Transformation')
    plt.xlabel(col)
    plt.ylabel('Frequency')

    # Perform polynomial feature transformation
    poly = PolynomialFeatures(degree=2, include_bias=False)
    transformed_col = poly.fit_transform(df[col].values.reshape(-1, 1))

    # Plot histograms after transformation
    plt.subplot(len(column_names), 3, i * 3 + 2)
    plt.hist(transformed_col[:, 0], bins=20, color='green', alpha=0.7)
    plt.title(f'{col} - After Transformation (Feature 1)')
    plt.xlabel(f'{col}_poly_1')
    plt.ylabel('Frequency')

    plt.subplot(len(column_names), 3, i * 3 + 3)
    plt.hist(transformed_col[:, 1], bins=20, color='orange', alpha=0.7)
    plt.title(f'{col} - After Transformation (Feature 2)')
    plt.xlabel(f'{col}_poly_2')
    plt.ylabel('Frequency')

    # Update the DataFrame with transformed data
    df[f'{col}_poly_1'] = transformed_col[:, 0]

    # Display transformed data
    transformed_data = pd.DataFrame({f'{col}_poly_1': transformed_col[:, 0]})
    f.save_table(transformed_data)

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
f.save_table(df)
# Show the plots
f.save_image(plt)

