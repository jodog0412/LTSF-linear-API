# LTSF-Linear-API
## Description
API for using `LTSF-Linear` easily.
## What is the `LTSF-Linear`?
https://github.com/cure-lab/LTSF-Linear   
`LTSF-Linear` is the SOTA of the long-time series forecasting model.  
And this is from the paper ["Are Transformers Effective for Time Series Forecasting?"](https://arxiv.org/abs/2205.13504, "arxiv")
## Why is the 'LTSF-Linear'?
* __High accessibility__  
It is deep-learning model, but very efficient.  
So, __you can use this model without expensive GPU.__
* __High Performance__  
<img src="https://user-images.githubusercontent.com/83653380/231390619-4fe2b936-99e0-469f-bba8-50dd5ac431b4.png" width="75%" height="55%" title="ETTH1 benchmark"></img>  
(You can check this benchmark in [this site](https://paperswithcode.com/sota/time-series-forecasting-on-etth1-720))  
It recorded the __highest performance__ on time-series-prediction.  

## Usage Examples
### 📈 Stock price prediction
* __code__  
```python
from yahooquery import Ticker
raw=Ticker('AAPL').history(period='2y').xs('AAPL')
window_size=122
forecast_size=30

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
```
* __Result__  
<img src="https://user-images.githubusercontent.com/83653380/231395184-d6a119bc-a427-4fc0-9826-cd7a17ace163.png" width="80%" height="60%" title="stock price prediciton"></img>  
