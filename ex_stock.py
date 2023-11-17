
from function import *
import pandas as pd

#Usage example 1
from yahooquery import Ticker
raw=Ticker('AAPL').history(period='1y').xs('AAPL')
window_size=28
forecast_size=7
date, data=targetParsing(raw,'adjclose') # preprocess raw data
dataloader=customDataLoader(data,
                            window_size,
                            forecast_size,
                            batch_size=4) #make dataloader
pred=trainer(data,
             dataloader,
             window_size,
             forecast_size).implement() #train and evaluate
figureplot(date,data,pred,window_size,forecast_size) #plot the result.


