import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


#title
app_name= 'Stock Market Forecasting App'
st.title(app_name)
st.subheader('This app is crated to forcast the stock market price of the selected company')

#add image from online resource
st.image("image1.jpg")

#take input from the user of app about the start and end date
#sidebar
st.sidebar.header('Select the parameters from below:')

start_date = st.sidebar.date_input('Start date', date(2020,1,1))
end_date = st.sidebar.date_input('End date', date(2020,12,31))

# add ticker symbol list
ticker_list = ['AAPL','TSLA','IONQ','META','GOOG','MSFT','NFLX']
ticker = st.sidebar.selectbox('Select the Company: ', ticker_list)

#fetch data from user inputs using yfinance library

data= yf.download(ticker, start=start_date, end= end_date)

#add date as a column to the dataframe
data.insert(0, "Date", data.index, True)
data.reset_index(drop=True, inplace=True)
st.write('Data from ', start_date, 'to ',end_date)
data[0]= ticker
st.write(data)

#plot the data
st.header("Data Visualization")
st.subheader("Plot of the data")

data.columns = [col[0] for col in data.columns]  # keep only first level (like 'Date', 'Close', etc.)

# Drop unwanted column '0' if it exists
if '0' in data.columns:
    data = data.drop(columns=['0'])

# Plot
fig = px.line(data, x='Date', y='Close', title='Closing price of the stock')
st.plotly_chart(fig)

#add a select box to select column from data
column = st.selectbox("Select the column to be used for forecasting", data.columns[1:])

#subsetting the data
data= data[['Date', column]]
st.write('Selected Data')
st.write(data)

#ADF test chech stationarity
st.header('Is data Stationary?')
st.write(adfuller(data[column])[1]<0.05 )

#lets decompose the data
st.header('Decomposition of the data')
decomposition = seasonal_decompose(data[column], model= 'additive', period=12)
st. write(decomposition.plot())

#make same plot in plotly
st.write("##Plotting the decomposition in plotly")
st.plotly_chart(px.line(x=data["Date"], y= decomposition.trend, title = 'Trend', width= 1200, height=400, labels={'x': 'Date','y':'Price'}).update_traces(line_color='Yellow'))
st.plotly_chart(px.line(x=data["Date"], y= decomposition.seasonal, title = 'Seasonality', width= 1200, height=400, labels={'x': 'Date','y':'Price'}).update_traces(line_color='Red'))
st.plotly_chart(px.line(x=data["Date"], y= decomposition.resid, title = 'Residuals', width= 1200, height=400, labels={'x': 'Date','y':'Price'}).update_traces(line_color='Green', line_dash='dot'))

#lets run the model
#user input for three parameters of the model and seasonal order
p = st.slider('Select the value of p ', 0, 5, 2)
d = st.slider('Select the value of d ', 0, 5, 1)
q = st.slider('Select the value of q ', 0, 5, 2)
seasonal_order = st.number_input('Select the value of seasonal p ', 0, 24, 12)

#train model
model= sm.tsa.statespace.SARIMAX(data[column], order=(p,d,q), seasonal_order=(p,d,q,seasonal_order))
model = model.fit()

#print model summary
st.header('Model Summary')
st.write(model.summary())
st.write('---------')

#predict the values with user input values
st.write("<p style='color:green; font-size:50px; font-weight: bold;'> Forecasting the data</p>", unsafe_allow_html=True)
forecast_period= st.number_input('##Enter forecast period in days', value=10)

#predict all the value for the forecast period and the current dataset
predictions = model.get_prediction(start=len(data), end= len(data)+forecast_period-1)
predictions = predictions.predicted_mean
st.write(predictions)

#add index to the predications
predictions.index= pd.date_range(start=end_date, periods=len(predictions), freq='D')
predictions = pd.DataFrame(predictions)
predictions.insert(0, "Date", predictions.index, True)
st.write("Predictions", predictions)
st.write("Actual Data", data)

#lets plot the data
fig = go.Figure()
#add actual data to the plot
fig.add_trace(go.Scatter(x=data["Date"], y= data[column], mode='lines', name= 'Actual', line=dict(color='green')))
#add predicted data to the plot
fig.add_trace(go.Scatter(x=predictions["Date"], y= predictions["predicted_mean"], mode='lines', name= 'Predicted', line=dict(color='yellow')))
#set the title and axis labels
fig.update_layout(title= 'Actual vs Predicted', xaxis_title='Date', yaxis_title="Price", width= 1000 ,height= 400) 
#display the plot
st.plotly_chart(fig)


#add buttons to show and hide separate plots
show_plots= False
if st.button("Show separate plots"):
    if not show_plots:
        st.write(px.line(x=data["Date"], y=data[column], title='Actual', width= 1200, height= 400, labels={'x':'Date','y':'Price'}).update_traces(line_color='Yellow'))
        st.write(px.line(x=predictions["Date"], y= predictions["predicted_mean"], title= "Predicted", width= 1200, height= 400, labels={'x':'Date','y':'Price'}).update_traces(line_color='Green'))
        show_plots=True
    else:
        show_plots=False
    
#add hid plot button
hide_plots=False
if st.button("Hide separate plots"):
    if not hide_plots:
        hide_plots= True
    else:
        hide_plots=False

st.write("-----")
st.write("About the author:")
st.write("<p style='color:green; font-size:50px; font-weight: bold;'> Dr. Mohd Rashid</p>", unsafe_allow_html=True)
