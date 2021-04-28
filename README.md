# Practicum II: Reddit Stock Sentiment Analysis

## Project Summary

The goal of this project is to understand whether sentiment of stock comments on Reddit can be used to predict future stock prices. 

The analysis will consist of applying sentiment scores to each day of comment activity on specific subreddits, then comparing that data to stock market prices. 

I will then attempt to predict next-day prices in order to see if there are any trends that can be predicted by sentiment. 



### Libraries 

The primary library for cleaning and structuring data was Pandas. 

Requests and Praw were used to pull reddit data. 

FFN was used to pull stock data. 

Keras was used for the LSTM model.  

```
#Import Libraries
import sys
import pandas as pd
import json
import datetime
import re #regex
import requests #APIs
import praw #reddit API enhanced library
import ffn #stock data
from textblob import TextBlob
import nltk
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import tensorflow as tf
from tensorflow import keras #lstm
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sn
import numpy as np
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import pickle

#!{sys.executable} -m pip install praw
#!{sys.executable} -m pip install textblob
#!{sys.executable} -m pip install ffn
#!{sys.executable} -m pip install tensorflow
```

## Data

### Reddit Posts and Comments

One of the primary pieces of this project was to pull text data from reddit at both the comment and the post level. These were used separately, and from 2 different subreddits, to compare their effectiveness in predicting stock price.

```
# function to get data from pushshift api
def getPushshiftData(query, after, before, sub):
    url = 'https://api.pushshift.io/reddit/search/submission/?title='+str(query)+'&size=1000&after='+str(after)+'&before='+str(before)+'&subreddit='+str(sub)
    print(url)
    r = requests.get(url)
    data = json.loads(r.text)
    return data['data']
    
# get relevant data from data extracted using previous function
def collectSubData(subm):
    subData = [subm['id'], subm['title'], subm['url'], datetime.datetime.fromtimestamp(subm['created_utc']).date()]
    try:
        flair = subm['link_flair_text']
    except KeyError:
        flair = "NaN"
    subData.append(flair)
    subStats.append(subData)
    
```

When running the above functions, I looked at the following subreddits:
  1. r/Stocks
  2. r/WallStreetBets
  
  
For post data, I pulled all posts as far back as I could. This process took a while, but was not as taxing as pulling all comments. 

```
while len(data) > 0:
    for submission in data:
        collectSubData(submission)
        subCount+=1
    # Calls getPushshiftData() with the created date of the last submission
    print(len(data))
    print(str(datetime.datetime.fromtimestamp(data[-1]['created_utc'])))
    after = data[-1]['created_utc']
    data = getPushshiftData(query, after, before, sub)
```

For comments data, I pulled all text for the Daily Discussion Threads in each subreddit. This kept the data pull from being so large that I could not run it in a reasonable amount of time, and it was also a way to trim down the data to more relevant discussions for sentiment, as opposed to posts that may be popular but are used more as jokes.

```
#collect stocks comments using praw
comments_by_day_stocks=[]
for url in df_stocks['url'].tolist():
    submission = reddit_api.submission(url=url)
    submission.comments.replace_more(limit=0)
    comments=list([(comment.body) for comment in submission.comments])
    comments_by_day_stocks.append(comments)
```

These limitations on the data kept each of the files from being out of control in terms of size, as they were all under 250 MB.

## Sentiment Analysis 

In order to apply sentiment scores to the text data in the comments, I used the VADER library, which stands for Valence Aware Dictionary for Sentiment Reasoning.

VADER looks at Polarity in the text, which applies positive and negative scores, and it takes into account Intensity, or emotional strength, in its scores.

```
## run vader sentiment analyzer for stocks

scores=[]
for comments in comments_by_day_stocks:
    sentiment_score=0
    try:
        for comment in comments:
            sentiment_score=sentiment_score+analyser.polarity_scores(comment)['compound']
    except TypeError:
        sentiment_score=0
    
    scores.append(sentiment_score)
    
```

I took a different approach for the post title data, instead looking at the individual words and finding instances of strong buy (bear) or sell (bull) indicators. 

```
# Define bull and bear terms
bull_words=['call', 'long', 'all in', 'moon', 'going up', 'rocket', 'buy', 'long term', 'green']
bear_words=['put', 'short', 'going down', 'drop', 'bear', 'sell', 'red']
```

```
# Apply scores for times keywords appear in titles
for title in titles:
    bull=False
    bear=False
    for word in bull_words:
        if word in title:
            bull=True
    if re.findall(r'(\b\d{1,4}[c]\b)|(\b\d{1,4}[ ][c]\b)', title):
            bull=True
    for word in bear_words:
        if word in title:
            bear=True
    if re.findall(r'(\b\d{1,4}[p]\b)|(\b\d{1,4}[ ][p]\b)', title):
            bear=True 
    if bull==True and bear==True:
        bull_scores.append(0)
        bear_scores.append(0)
    if bull==False and bear==False:
        bull_scores.append(0)
        bear_scores.append(0)
    if bull==True and bear==False:
        bull_scores.append(1)
        bear_scores.append(0)
    if bull==False and bear==True:
        bull_scores.append(0)
        bear_scores.append(1)
```

Looking at each data set, certain trends pop out when looking at commonly used words:

![image](https://user-images.githubusercontent.com/1417344/116335184-a712f280-a793-11eb-87f5-571f4e7c3774.png)


![image](https://user-images.githubusercontent.com/1417344/116335214-b2feb480-a793-11eb-816b-4c4adb1c2c84.png)




```
# Generate the image
wordcloud = WordCloud(stopwords=stopwords, background_color="white", max_words=100, min_word_length=5).generate(text)

# visualize the image
fig=plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('r/Stocks Positive Titles Word Cloud')
plt.show()
```

## Stock Data

For comparison to stock prices, I pulled data for the SPY ETF, which tracks the S&P 500 index. This provided a broad market perspective to compare to sentiment, which I found ideal since I was not trying to compare to a single stock price. 

```
# spy price pulled from date reddit data becomes available

spy=ffn.get('spy', start='2017-01-01')
```


```
# Merge datasets
df_stocks = pd.merge(df_stocks,spy,left_on='date',right_on='Date',how='inner')
```



Plotting the sentiment data with stock price is initially too erratic to draw relationships.
```
df_wsb[['spy','sentiment score']].plot(secondary_y='sentiment score', figsize=(16, 10))
```
![image](https://user-images.githubusercontent.com/1417344/116335045-6dda8280-a793-11eb-9a47-9bc83e66e4da.png)




Fourier Transformations were used to smooth out the sentiment data. Fourier Transform is a time-based transformation, so it can identify cycles in data, so it works well in smoothing out the spikey sentiment data while keeping the underlying patterns.

```
close_fft = np.fft.fft(np.asarray(df_wsb['sentiment score'].tolist()))
fft_df = pd.DataFrame({'fft':close_fft})
fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
fft_list = np.asarray(fft_df['fft'].tolist())

for num_ in [5, 10, 15, 20]:
    fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
    df_wsb['fourier '+str(num_)]=np.fft.ifft(fft_list_m10)
```

Sentiment scores for the comments as well as both bull and bear scores were compared to stock prices. Typically, before, during, and after a strong bull period or strong bear period, you can see drastic swings in bear and bull sentiment.

![image](https://user-images.githubusercontent.com/1417344/116335283-d45fa080-a793-11eb-853c-bee68d080dab.png)



![image](https://user-images.githubusercontent.com/1417344/116335274-d0cc1980-a793-11eb-9e0f-77a6a64f19f3.png)





## LSTM Model

The initial EDA around picking inputs for the model was established from a correlation matrix:

```
# Stocks Comments Correlation
corrMatrix = df_stocks.corr()

sn.heatmap(corrMatrix, annot=True)
plt.show()

```

![image](https://user-images.githubusercontent.com/1417344/116334626-c3faf600-a792-11eb-8f62-d9e62b3f3b19.png)


The model performance was pretty consistent regardless of using all input columns or just using the top performing ones, but when limiting to the top correlated fields, the model ran much more quickly.

The key differences between each model were the data sources used. When comparing the out-of-the-box methods on each dataset, the Stocks Keywords model performed the worst, followed by the WSB Comments model, then the WSB Keywords model, and finally the best performing iteration was the Stocks Comments at 14% Mean Absolute Error. After the best performing dataset was selected, I ran through several iterations of the columns used and settled on just using the fields that were most correlated with stock price, because that model performed about twice as well and ran more quickly too. Then, I fine-tuned the model parameters to get the best performing result.

![image](https://user-images.githubusercontent.com/1417344/116335410-11c42e00-a794-11eb-9967-fb5dcd293fd4.png)



The model creation process included splitting the data into training and test sets, identifying the window frame to use, looking at the gap for predicting the next day, and converting the datasets into arrays. Then, the model was defined using the Keras library. Several iterations of model specifications were tested. The top performing model had a 14 day lookback window, 3 LSTM layers, and 3 Dense layers. 

![image](https://user-images.githubusercontent.com/1417344/116334757-f7d61b80-a792-11eb-8a71-8fe1a9edd4bf.png)


```
# get relevant columns and divide into train and test sets
df=df_stocks[['norm_price', 'norm_sentiment', 'norm_fourier5', 'norm_fourier10', 
           'norm_fourier15', 'norm_fourier20']].to_numpy()
window=30
gap=1
data=[]
for x in range(len(df)-window): 
    data.append(df[x:x+window])
data=np.asarray(data)
train=data[:-50]
test=data[-50:]
np.random.shuffle(train)

#Train test
X_train=[]
y_train=[]
for d in train:
    X_train.append(remove_first(d[:window-gap]))
    y_train.append(d[-1][0])

X_test=[]
y_test=[]
for d in test:
    X_test.append(remove_first(d[:window-gap]))
    y_test.append(d[-1][0])
    
X_train=np.asarray(X_train)
y_train=np.asarray(y_train)
X_test=np.asarray(X_test)
y_test=np.asarray(y_test)

print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('X_test shape: ', X_test.shape)
print('y_test shape: ', y_test.shape)

tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

mc=tf.keras.callbacks.ModelCheckpoint(filepath='lstm_comment_sentiment_stocks_1.h5', monitor='val_loss', save_best_only=True)
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(96, return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2])),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.LSTM(96,return_sequences=True),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.LSTM(96),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(48, activation="relu"),
    tf.keras.layers.Dense(24, activation="relu"),
    tf.keras.layers.Dense(12, activation="relu"),
    tf.keras.layers.Dense(1),
])

model.compile(optimizer='adam', loss='mean_squared_error')#, metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=32, validation_data= (X_test, y_test), epochs=250, callbacks=[mc]).history

```


## ARIMA Model

To create a comparison to the LSTM models, I also developed an ARIMA model to determine how a univariate time series would perform. Since the dataset was simpler with just time and stock price, the model development was a much smoother process.

The initial step was to test for stationarity, followed by looking at seasonality and trend. 

```
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    plt.plot(timeseries, color='yellow',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)
    print("Results of dickey fuller test")
    adft = adfuller(timeseries,autolag='AIC')
    # output for dft will give us without defining what the values are.
    #hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)

test_stationarity(spy['spy'])
```
![image](https://user-images.githubusercontent.com/1417344/116335441-21437700-a794-11eb-8108-b327e9222ea6.png)

Since the p-value is > 0.05, we accept the null hypothesis and conclude that the data is non-stationary. 


```
result = seasonal_decompose(spy, model='multiplicative', freq = 30)
fig = plt.figure()  
fig = result.plot()  
fig.set_size_inches(16, 9)
```
![image](https://user-images.githubusercontent.com/1417344/116335477-2f919300-a794-11eb-8c13-ffec51b11ebc.png)



Taking the log of the spy data was sufficient for solving the stationarity problem in the data.

The model definition was relatively straightforward, since once again this was univariate instead of multivariate. 

```
# Define model and print summary
model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
test='adf',       # use adftest to find optimal 'd'
max_p=3, max_q=3, # maximum p and q
m=1,              # frequency of series
d=None,           # let model determine 'd'
seasonal=False,   # No Seasonality
start_P=0, 
D=0, 
trace=True,
error_action='ignore',  
suppress_warnings=True, 
stepwise=True)
print(model_autoARIMA.summary())
```

```
# Model Diagnostics
model_autoARIMA.plot_diagnostics(figsize=(15,8))
plt.show()
```
![image](https://user-images.githubusercontent.com/1417344/116335529-420bcc80-a794-11eb-82be-ae9de1f3c80e.png)



Plotting the data showcases how the data performs compared to historical actuals as well as direct comparison against actuals:

![image](https://user-images.githubusercontent.com/1417344/116335573-53ed6f80-a794-11eb-95c3-ca78e0b6b495.png)



```
fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train_data, label='training')
plt.plot(test_data, color = 'blue', label='Actual Stock Price')
plt.plot(fc_series, color = 'orange',label='Predicted Stock Price')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.10)
plt.title('SPY Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Actual Stock Price')
plt.legend(loc='upper left', fontsize=8)
plt.show()
```

The Mean Absolute Error of this model was right in line with the second-best performing LSTM model, but regardless the top-performing LSTM model was an improvement over using the above univariate approach. 


Finally, I created an evaluation of the results in order to compare how performance in the general market using dollar-cost-averaging would be with dates selected by this ARIMA model. 

```
#Convert forcasted data and test data to dataframes, then merge into one dataframe
fc_series = pd.DataFrame(fc_series)
test_data = pd.DataFrame(test_data)
df_pred = pd.merge(test_data,fc_series,left_on='Date',right_on='Date',how='inner')

#Rename columns
df_pred.columns.values[0] = "spy_actual"
df_pred.columns.values[1] = "spy_pred"

df_pred = pd.DataFrame(df_pred)
df_pred.reset_index(drop=False, inplace=True)

#Add weekday column for aggregation
df_pred["weekday"] = pd.to_datetime(df_pred.Date).dt.dayofweek

#Minimum predicted stock prices per week
min_week_values = df_pred.groupby([pd.Grouper(key='Date', freq='W-MON')])['spy_pred'].min()
#Source: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html

#Dataset for all of the predicted low stock price dates per week
min_dates = pd.merge(df_pred,min_week_values,left_on='spy_pred',right_on='spy_pred',how='inner')

#Number of purchase days
num_days = min_dates['Date'].count()

#Pick random dates to purchase for comparison
rand_dates = df_pred.sample(n = num_days)

#Cost using predicted purchase dates
pred_cost = (min_dates['spy_actual'] * 10).sum()

#Cost using random purchase dates
rand_cost = (rand_dates['spy_actual'] * 10).sum()
```

The result of this methodology was an improvement of around 0.3% when using the ARIMA model to identify optimal dates to buy in a dollar-cost-averaging strategy. While this seems small, a 0.3% improvement in the stock market could be huge if applied across multiple time periods and large sums of money. 

## Conclusion 

The primary finding of this analysis was that sentiment data can, in fact, be used to predict stock price movement with relatively high accuracy. The stock prices themselves weren't predicted, since the model was designed around minimizing error as opposed to identifying price exactly. Either way, the sentiment score model without a heavy amount of manipulationw as able to have an error around 6%, which I consider very good, and the multivariate LSTM model outperformed the univariate ARIMA model, indicating that sentiment data is at the very least impactful in the prediction of stock price movement.

While I would not recommend creating an entire investing strategy based on this model, I believe I confirmed what I set out to discover: that sentiment can be utilized as a factor in predicting stock price movement. In order to effectively deploy this method, I would suggest using it as a piece of a more complex model, not as the entire model input itself.

Next steps in this analysis would be to use Named Entity Recognition to identify specific stocks that are being discussed and see if that level of granularity can be predicted, compared to the overall market. 



## Sources 

https://www.reddit.com/dev/api/

https://praw.readthedocs.io/en/latest/index.html

https://medium.com/analytics-vidhya/sentiment-analysis-for-trading-with-reddit-text-data-73729c931d01

https://stackabuse.com/reading-and-writing-lists-to-a-file-in-python/

https://stackoverflow.com/questions/20490274/how-to-reset-index-in-a-pandas-dataframe

https://towardsdatascience.com/ner-for-extracting-stock-mentions-on-reddit-aa604e577be

https://www.geeksforgeeks.org/python-get-unique-values-list/

https://realpython.com/python-nltk-sentiment-analysis/#selecting-useful-features

https://stackoverflow.com/questions/58435657/how-to-access-column-after-pandas-groupby

https://quantdare.com/introduction-to-nlp-sentiment-analysis-and-wordclouds/

https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef7664pol

https://stackoverflow.com/questions/20490274/how-to-reset-index-in-a-pandas-dataframe

https://keras.io/api/layers/activations/

https://datatofish.com/correlation-matrix-pandas/

https://stackoverflow.com/questions/36367986/how-to-make-inline-plots-in-jupyter-notebook-larger

https://stackoverflow.com/questions/63427628/tensorflow-throws-me-the-error-valueerror-layer-sequential-expects-1-inputs-bu

https://stackoverflow.com/questions/45632549/why-is-the-accuracy-for-my-keras-model-always-0-when-training

https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html

https://www.analyticsvidhya.com/blog/2020/11/stock-market-price-trend-prediction-using-time-series-forecasting/

https://medium.com/human-in-a-machine-world/mae-and-rmse-which-metric-is-better-e60ac3bde13d
