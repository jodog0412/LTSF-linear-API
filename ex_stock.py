from function import *
import pandas as pd
from yahooquery import Ticker

raw=Ticker('AAPL').history(period='1y').xs('AAPL')
window_size,forecast_size=30,10
''' 1. preprocess raw data '''
date, data=split_data(raw,'adjclose')
''' 2. build dataloader '''
dataloader=build_dataLoader(data,
                            window_size=window_size,
                            forecast_size=forecast_size,
                            batch_size=4)
''' 3. train and evaluate '''
pred=trainer(data,
             dataloader,
             window_size=window_size,
             forecast_size=forecast_size).implement() 
''' 4. plot the result '''
figureplot(date,data,pred,window_size,forecast_size)
