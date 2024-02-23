import pandas as pd
from Console.package import Functions
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Create an instance of the Functions class
f = Functions()

# Import data using get_args
df = f.get_args("df")
target_col = f.get_args("target_col")
time_col = f.get_args("time_col")
fromm = f.get_args("from")
to = f.get_args("to")

# Specify the target variable and time column
target_var = target_col
time_column = time_col  # Replace with your actual time column name

# Filter data based on the specified time range
time_from = fromm  # Replace with your desired start date
time_to = to    # Replace with your desired end date

# Convert the time column to datetime format with the specified format
df[time_column] = pd.to_datetime(df[time_column], format="%d-%m-%Y", dayfirst=True)

# Set the DataFrame index to the time column
df = df.set_index(time_column)

# Sort the DataFrame by the index
df = df.sort_index()



# Train ARIMA model for the specified target variable
order = (2, 1, 1)  # Adjust the order as needed
model = sm.tsa.ARIMA(df[target_var], order=order)
res = model.fit()

# Forecast using the trained ARIMA model
forecast_values = res.predict(start=time_from, end=time_to, dynamic=False)

# Calculate Mean Absolute Error (MAE)
actual_values = df[target_var][time_from:time_to]
mae = abs(actual_values - forecast_values).mean()
print(f"Mean Absolute Error (MAE): {mae}")

# Plot a graph of actual vs. forecast values
plt.figure(figsize=(10, 6))
plt.plot(actual_values, label='Actual Values')
plt.plot(forecast_values, label='Forecast Values', linestyle='--')
plt.title(f"ARIMA Forecast for {target_var}")
plt.xlabel("Time")
plt.ylabel(target_var)
plt.legend()
f.save_image(plt)