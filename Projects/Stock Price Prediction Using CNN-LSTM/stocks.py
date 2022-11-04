import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn import preprocessing
from keras.layers import Conv1D,Flatten,MaxPooling1D,Bidirectional,LSTM,Dropout,TimeDistributed,MaxPool2D
from keras.layers import Dense,GlobalAveragePooling2D
import matplotlib.pyplot as plt
import os
import pprint
import tensorflow as tf

#install the yfinance library to directly get the stock prices lively if not installed
pip install yfinance

import yfinance as yf
import seaborn as sns

#give the ticker value a stock name you wish to forecast
ticker = 'AAPL'
stock = yf.download(tickers=ticker, period='5y', interval='1d')

#duplicate the stock copy 
stock2 = stock.copy()

import plotly.graph_objs as go
fig = go.Figure()

#get the candle stick pattern of the stock

fig.add_trace(go.Candlestick(x=stock.index,
                open=stock['Open'],
                high=stock['High'],
                low=stock['Low'],
                close=stock['Close'], name = 'market data'))

fig.update_layout(
    title='Live share price evolution',
    yaxis_title='Stock Price (USD per Shares)')

#plot the scatter graph of open high and close prices in a single layout

import plotly
import plotly.express as px
import plotly.graph_objects as go
stock2.reset_index(inplace=True)
trace1 = go.Scatter(
 x = stock2['Date'],
 y = stock2['Low'],
 mode = 'lines',
 name = 'low'
)
trace2 = go.Scatter(
 x = stock2['Date'],
 y = stock2['High'],
 mode = 'lines',
 name = 'high'
)
trace3 = go.Scatter(
 x = stock2['Date'],
 y = stock2['Open'],
 mode = 'lines',
 name = 'open'
)
layout = go.Layout(
 title = 'Stock Prices Plot over time  || Date vs Price',
 xaxis = {'title' : 'date'},
 yaxis = {'title' : 'Prices Predicted'}
)
fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
fig.show()

window_size = 50
week = 7
X = []
Y = []

for i in range(0 , len(stock) - window_size - 1 , 1):
    first = stock.iloc[i, 4]
    temp = []
    temp2 = []
    for j in range(window_size):
        temp.append((stock.iloc[i + j, 4] - first) / first)
    # for j in range(week):
    temp2.append((stock.iloc[i +window_size, 4] - first) / first)
    # X.append(np.array(stock.iloc[i:i+window_size,4]).reshape(50,1))
    # Y.append(np.array(stock.iloc[i+window_size,4]).reshape(1,1))
    # print(stock2.iloc[i:i+window_size,4])
    X.append(np.array(temp).reshape(50, 1))
    Y.append(np.array(temp2).reshape(1,1))

train_X,test_X,train_label,test_label = train_test_split(X, Y, test_size=0.2,shuffle=False)
len_t = len(train_X)

train_X = np.array(train_X)
test_X = np.array(test_X)
train_label = np.array(train_label)
test_label = np.array(test_label)

train_X = train_X.reshape(train_X.shape[0],1,50,1)
test_X = test_X.reshape(test_X.shape[0],1,50,1)

# Building our CNN-LSTM model 

model = Sequential()
#add model layers
model.add(TimeDistributed(Conv1D(128, kernel_size=5, activation='relu', input_shape=(None,50,1))))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(256, kernel_size=3, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(512, kernel_size=1, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Flatten()))
model.add(Bidirectional(LSTM(200,return_sequences=True)))
model.add(Dropout(0.25))
model.add(Bidirectional(LSTM(200,return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse')

# Training the model
model.fit(train_X, train_label, validation_data=(test_X,test_label), epochs=50,batch_size=64,shuffle =False)

predicted  = model.predict(test_X)
test_label = (test_label[:,0])
predicted = np.array(predicted[:,0]).reshape(-1,1)

df1 = pd.DataFrame(test_label)
df2 = pd.DataFrame(predicted)

#obtain the correlation between the test labels and predicted values
correlation = df1[0].corr(df2[0])
correlation

#Find out the R score value

for j in range(len_t , len_t + len(test_X)):
    temp =stock2.iloc[j,4]
    test_label[j - len_t] = test_label[j - len_t] * temp + temp
    predicted[j - len_t] = predicted[j - len_t] * temp + temp
    
import seaborn as sns
sns.regplot(x=test_label, y=predicted).set(title='R = '+str(round(correlation,2)))

#Getting the graph of original price and predicted price 

plt.plot(date,test_label, color = 'blue', label = ' Stock Price')
plt.plot(date,predicted, color = 'red', label = 'Predicted  Stock Price')
plt.title('Price Prediction of '+ticker)
plt.xlabel('Date')
plt.ylabel(ticker+' Price')
#plt.xticks('date')
plt.legend()
plt.show()


