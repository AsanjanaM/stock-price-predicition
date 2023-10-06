import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import yfinance as yf
import matplotlib as mp
from plotly import graph_objs as go





# Set the page title
st.title("Stock Price Prediction App")#

# Get user input for the stock symbol and number of days for prediction

option = st.sidebar.selectbox('Select one symbol', ( 'AAPL', 'MSFT',"SPY",'WMT','GME','MU','NFLX','BNOX'))

import datetime

today = datetime.date.today()
before = today - datetime.timedelta(days=8395)
start_date = st.sidebar.date_input('Start date', before)
end_date = st.sidebar.date_input('End date', today)

# Fetch the stock data using yfinance
stock_data = yf.download(option, start='2010-01-01', end='2023-10-06')

# Prepare the data for training
stock_data['Date'] = stock_data.index
stock_data['Date'] = stock_data['Date'].apply(lambda x: x.toordinal())
X = stock_data[['Date']]
y = stock_data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
prediction_days = 45  # Set the number of days for prediction
prediction_dates = pd.date_range(start=stock_data.index[-1], periods=prediction_days+1, closed='right')
prediction_dates = prediction_dates[1:]  # Exclude the last date from the range
prediction_dates_ord = prediction_dates.to_series().apply(lambda x: x.toordinal())
predictions = model.predict(prediction_dates_ord.values.reshape(-1, 1))


# Display the actual stock prices
st.subheader("Actual Stock Prices:")
st.line_chart(stock_data['Close'])

# Display the predicted stock prices
st.subheader("Predicted Stock Prices:")
predicted_data = pd.DataFrame({'Date': prediction_dates, 'Close': predictions})
st.line_chart(predicted_data.set_index('Date'))

# Display the actual and predicted stock prices together
st.subheader("Actual vs. Predicted Stock Prices:")
combined_data = pd.concat([stock_data['Close'], predicted_data.set_index('Date')['Close']], axis=1)
combined_data.columns = ['Actual', 'Predicted']
st.line_chart(combined_data)



