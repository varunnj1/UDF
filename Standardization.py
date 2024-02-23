"""import pandas as pd
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

# Standardize each specified column
scaler = StandardScaler()
df_standardized = pd.DataFrame()

for col in column_names:
    standardized_col = scaler.fit_transform(df[col].values.reshape(-1, 1))
    df_standardized[col + "_standardized"] = standardized_col.flatten()


#print(df_standardized)
f.save_table(df_standardized)
"""
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
    axs[i, 0].hist(df[col].dropna(), bins=20, color='blue', alpha=0.7)
    axs[i, 0].set_title(f'{col} - Before Standardization')
    axs[i, 0].set_xlabel(col)
    axs[i, 0].set_ylabel('Frequency')

    # Standardize the column
    scaler = StandardScaler()
    standardized_col = scaler.fit_transform(df[col].values.reshape(-1, 1))

    # Plot histogram after standardization
    axs[i, 1].hist(standardized_col, bins=20, color='green', alpha=0.7)
    axs[i, 1].set_title(f'{col} - After Standardization')
    axs[i, 1].set_xlabel(f'{col}_standardized')
    axs[i, 1].set_ylabel('Frequency')

    # Update the DataFrame with standardized data
    df[f'{col}_standardized'] = standardized_col.flatten()
    # Display standardized data
    standardized_data = pd.DataFrame({f'{col}_standardized': scaler.fit_transform(df[col].values.reshape(-1, 1)).flatten() for col in column_names})

   # standardized_data = pd.DataFrame(standardized_col, columns=[f'{col}_standardized'])
   # print(f"Standardized Data for {col}:")
    #print(standardized_data)
    f.save_table(standardized_data)
# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Show the plots
#plt.show()
f.save_image(plt)

