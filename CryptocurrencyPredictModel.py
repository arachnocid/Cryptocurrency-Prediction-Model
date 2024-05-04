# Import necessary libraries
import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, Span, Label, LegendItem

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout

from statsmodels.tsa.statespace.varmax import VARMAX
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# 1. Sequential model
# Import data and perform data cleaning
data = pd.read_csv('Bitcoin Historical Data (2).csv')
#print(data)


def convert_to_numeric(value):
    """
    Convert a value to numeric format.
    If the value is already a float, returns the value as it is.
    If the value contains 'K', 'M', or 'B', converts it to a numeric format accordingly.
    :param value: the value to be converted
    :type value: str or float
    :return: the numeric value
    :rtype: float
    """
    if isinstance(value, float):
        return value
    elif 'K' in value:
        return float(re.sub(r'[^\d.]', '', value)) * 1000
    elif 'M' in value:
        return float(re.sub(r'[^\d.]', '', value)) * 1000000
    elif 'B' in value:
        return float(re.sub(r'[^\d.]', '', value)) * 1000000000
    else:
        return float(re.sub(r'[^\d.]', '', value))


data.replace({',': '', '%': ''}, regex=True, inplace=True)
data = data.applymap(convert_to_numeric)

data['Date'] = pd.to_datetime(data['Date'], format='%m%d%Y')
data = data.loc[data['Date'] <= '2024-03-24'].copy()

#print(data.sort_values(by='Date', inplace=False, ascending=False).head(15))

data['Date'] = data['Date'].interpolate()
data.sort_values(by='Date', inplace=True)

#print(data.sort_values(by='Date', inplace=False, ascending=False).head(15))

data['Date'] = data['Date'].astype('int64') / 10**9
#print(data)

for column in data.columns[1:]:
    data[column] = pd.to_numeric(data[column], errors='coerce')

for col in data.columns[1:]:
    data[col] = data[col].interpolate(method='linear', limit_direction='both')

print("Missing values:\n", data.isnull().sum())
print("NaN values present:", data.isna().values.any())

# Data visualization
max_length = len(data['Price'])

start_date = pd.to_datetime('8/14/2010')
dates = pd.date_range(start_date, periods=max_length)

p = figure(title='Bitcoin Price Prediction', x_axis_label='Date', y_axis_label='Price', width=1200, height=600, x_axis_type="datetime")

p.line(dates, data['Price'], legend_label='Actual Price', line_color='blue')

hover = HoverTool()
hover.tooltips = [('Date', '@x{%F}'), ('Price', '@y')]
hover.formatters = {'@x': 'datetime'}
p.add_tools(hover)
show(p)

# Scaling and splitting data into training and testing dataset
features = data.drop('Price', axis=1)
target = data['Price']

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

target_scaled = pd.Series.to_numpy(target)
target_scaled = target_scaled.reshape(-1, 1)
target_scaled = scaler.fit_transform(target_scaled)

print("Original Shapes - Features:", features_scaled.shape, "Target:", target_scaled.shape)

x_train, x_test, y_train, y_test = train_test_split(features_scaled, target_scaled, random_state=5)

target_scaler = MinMaxScaler()
y_train_scaled = target_scaler.fit_transform(y_train)
y_test_scaled = target_scaler.transform(y_test)

# Model compilation
# The MAE should be <= 0.01; otherwise, the accuracy will be compromised
# If the currency rate prediction shows unsatisfactory results, recompile the model
model = Sequential()
model.add(Dense(units=32, activation='relu', use_bias=True, bias_initializer='zeros'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu', use_bias=True, bias_initializer='zeros'))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

history = model.fit(x_train, y_train, epochs=21, batch_size=32, validation_split=0.3)

loss, mae = model.evaluate(x_test, y_test)
print(f'Test Loss: {loss}, Test MAE: {mae}')

predicted_prices = model.predict(x_test)

# A comparison visualization of the original prices and those predicted by the model
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Price', color='blue')
plt.plot(predicted_prices, label='Predicted Price', color='red', alpha=0.9, linestyle='--')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# 2. Prediction for a specific time interval ahead - VARMAX
# Data preparation
warnings.simplefilter('ignore', ConvergenceWarning)

data_scaled = scaler.fit_transform(data.drop('Price', axis=1))
print("Data:", data_scaled.shape)

# Creating a VARMAX model and predicting future values for each feature
future_steps = 100 # can be replaced by any other period, but the model will always predict the terminating form of the exchange rate
                   # In the chart there is a line that separates the predicted terminal form of the exchange rate from the values that can be interpreted

varmax_predictions = []

varmax_model = VARMAX(data_scaled, order=(2, 1))
varmax_result = varmax_model.fit(disp=False)
future_forecast = varmax_result.forecast(steps=future_steps)
varmax_predictions.append(future_forecast)

# Convert VARMAX predicted values to an array
varmax_predictions = np.array(varmax_predictions).T

print("VARMAX Shapes - Features:", varmax_predictions.shape)

# Generated data reshaping and concatenation with original data
varmax_predictions_reshaped = varmax_predictions.reshape(future_steps, -1)

print("VARMAX Shapes - Features:", varmax_predictions_reshaped.shape)
print("Data Shape - Features:", data_scaled.shape)

features_with_varmax = np.concatenate((data_scaled, varmax_predictions_reshaped))
print("New Data Shape - Features:", features_with_varmax.shape)

# Predict future prices ahead
future_prices_scaled = model.predict(features_with_varmax)

# Features and prices unscaling
min_price = data['Price'].min()
max_price = data['Price'].max()

future_prices = future_prices_scaled * (max_price - min_price) + min_price
target_unscaled = target_scaled * (max_price - min_price) + min_price

# Interactive graph plot
max_length = max(len(target_unscaled), len(future_prices))

start_date = pd.to_datetime('8/14/2010')
dates = pd.date_range(start_date, periods=max_length)

# Rolling price for predicted prices
#rolling_window_size = 7
#rolling_mean_predicted_prices = np.convolve(future_prices[:max_length].flatten(), np.ones(rolling_window_size)/rolling_window_size, mode='valid')

p = figure(title='Bitcoin Price Prediction', x_axis_label='Date', y_axis_label='Price', width=1200, height=600, x_axis_type="datetime")

p.line(dates, target_unscaled[:max_length].flatten(), legend_label='Actual Price', line_color='blue')
p.line(dates, future_prices[:max_length].flatten(), legend_label='Predicted Price', line_color='red', line_dash='dashed')
#p.line(dates[rolling_window_size-1:], rolling_mean_predicted_prices, legend_label='Smoothed Predicted Price', line_color='black')

# Separates the predicted terminal form of the exchange rate from the values that can be interpreted
two_thirds_index = int(len(varmax_predictions_reshaped) * 2 / 3)
two_thirds_future_date = dates[-two_thirds_index]

tboundary = Span(location=two_thirds_future_date, dimension='height', line_color='black', line_width=1)
p.add_layout(tboundary)

label = Label(x=two_thirds_future_date, y=1, text='Terminal Prediction Boundary', text_font_size='10pt', text_color='black')
legend_item = LegendItem(label='Terminal Prediction Boundary')
p.legend.items.append(legend_item)

hover = HoverTool()
hover.tooltips = [('Date', '@x{%F}'), ('Price', '@y')]
hover.formatters = {'@x': 'datetime'}
p.add_tools(hover)
show(p)
