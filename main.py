
from function import *

#Usage example 1
from yahooquery import Ticker
raw=Ticker('AAPL').history(period='2y').xs('AAPL')
window_size=92
forecast_size=31

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

# Usage example 2
def predict(raw,window_size:int,forecast_size:int):
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

# predict(raw,window_size=93,forecast_size=31)