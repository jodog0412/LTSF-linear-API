
from function import *
import pandas as pd

#Usage example 1
from yahooquery import Ticker
raw=Ticker('AAPL').history(period='1y').xs('AAPL')
window_size,forecast_size=15,5
date, data=split_data(raw,'adjclose') # preprocess raw data
dataloader=build_dataLoader(data,
                            window_size=window_size,
                            forecast_size=forecast_size,
                            batch_size=4) #make dataloader
pred=trainer(data,
             dataloader,
             window_size=window_size,
             forecast_size=forecast_size).implement() #train and evaluate
figureplot(date,data,pred,window_size,forecast_size) #plot the result.


