import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
import streamlit as st
import matplotlib.pyplot as plt

data = pd.read_csv('Saudi Arabia Net Consumption.csv')

data['Unnamed: 0'] = pd.to_datetime(data['Unnamed: 0'])
data.set_index('Unnamed: 0', inplace=True)

st.sidebar.title("Settings")
forecast_years = st.sidebar.slider("Forecast Years", min_value=1, max_value=15, value=6)
st.title("Holt-Winters Forecasting App")

y_hat_avg = data.copy()
fit1 = ExponentialSmoothing(
    np.asarray(data['net consumption']),
    seasonal_periods=2,
    trend='multiplicative',
    seasonal='multiplicative'
).fit()

future_years = pd.date_range(start=data.index[-1], periods=forecast_years, freq='A')
future_df = pd.DataFrame(index=future_years)

forecast_horizon = len(future_years)
forecast_values = fit1.forecast(steps=forecast_horizon)

future_df['Holt_Winter_Forecast'] = forecast_values

plt.figure(figsize=(12, 6))
plt.plot(data['net consumption'], label='Historical Data')
plt.plot(future_df.index, future_df['Holt_Winter_Forecast'], label='Future Forecast', linestyle='--')
plt.title('Saudi Arabia Net Consumption Forecast')
plt.xlabel('Year')
plt.ylabel('Net Consumption')
plt.legend()
plt.grid(True)

st.pyplot(plt)
