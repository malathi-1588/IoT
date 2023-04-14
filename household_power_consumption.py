import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the data into a Pandas dataframe
df = pd.read_csv("household_power_consumption.csv", sep=";", parse_dates={"datetime": ["Date", "Time"]}, infer_datetime_format=True, low_memory=False, na_values=["?"])

columns = ['Global_active_power', 'Global_reactive_power',
       'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
       'Sub_metering_3']

# Replace missing values with NaN
df.replace('?', np.nan, inplace=True)

df.set_index("datetime", inplace=True)

# Resample the data to hourly frequency and forward-fill missing values
df = df.resample("H").ffill()

# Convert data types to float
for col in columns:
    df[col] = df[col].astype(float)
    
df = df.apply(pd.to_numeric, errors='coerce')
    
df["Global_active_power"].fillna(df["Global_active_power"].mean(), inplace=True)
df["Global_reactive_power"].fillna(df["Global_reactive_power"].mean(), inplace=True)
df["Voltage"].fillna(df["Voltage"].mean(), inplace=True)
df["Global_intensity"].fillna(df["Global_intensity"].mean(), inplace=True)
df["Sub_metering_1"].fillna(df["Sub_metering_1"].mean(), inplace=True)
df["Sub_metering_2"].fillna(df["Sub_metering_2"].mean(), inplace=True)
df["Sub_metering_3"].fillna(df["Sub_metering_3"].mean(), inplace=True)

# Visualize the time series data
df["Global_active_power"].plot(figsize=(50, 20))
plt.xlabel("Date")
plt.ylabel("Global Active Power (kilowatts)")
plt.show()

# Perform time series decomposition
decomp = seasonal_decompose(train["Global_active_power"], model="additive", period=24)
decomp.plot()
plt.show()

# Plot autocorrelation and partial autocorrelation
plot_acf(train["Global_active_power"], lags=48)
plt.show()
plot_pacf(train["Global_active_power"], lags=48)
plt.show()

# Define the SARIMA model
model = SARIMAX(ts, order=(2, 1, 2), seasonal_order=(0, 1, 1, 24), enforce_invertibility=False, enforce_stationarity=False)

# Fit the SARIMA model to the data
result = model.fit()

# Generate forecast for the next 24 hours
forecast = result.forecast(36)

# Visualize the forecasted values
ts.plot(figsize=(50, 10))
forecast.plot()
plt.xlabel("Date")
plt.ylabel("Global Active Power (kilowatts)")
plt.show()

# Print the model summary
print(result.summary())
