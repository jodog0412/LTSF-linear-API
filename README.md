# LTSF-Linear-API
## Description
API for using `LTSF-Linear` easily.
## ❓ What is the `LTSF-Linear`?
https://github.com/cure-lab/LTSF-Linear   
`LTSF-Linear` is the SOTA of the long-time series forecasting model.  
And this is from the paper ["Are Transformers Effective for Time Series Forecasting?"](https://arxiv.org/pdf/2205.13504.pdf, "arxiv")
## ✨ Why is the `LTSF-Linear`?
* __High accessibility__  
It is deep-learning model, but very efficient.  
So, __you can use this model without expensive GPU.__
* __High Performance__  
<img src="https://user-images.githubusercontent.com/83653380/231390619-4fe2b936-99e0-469f-bba8-50dd5ac431b4.png" width="75%" height="55%" title="ETTH1 benchmark"></img>  
(You can check this benchmark in [this site](https://paperswithcode.com/sota/time-series-forecasting-on-etth1-720))  
It recorded the __highest performance__ on time-series-prediction.  

## Usage Examples
### 📈 Stock price prediction
* __Code__  
```python
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
```
* __Data__  
<img src="https://user-images.githubusercontent.com/83653380/231967177-68f284a1-1b41-4fce-ab0b-6a563b3777d3.png" width="50%" height="38%" title="original stock price"></img>  

* __Prediction__  
<img src="https://user-images.githubusercontent.com/83653380/231395184-d6a119bc-a427-4fc0-9826-cd7a17ace163.png" width="80%" height="60%" title="stock price prediciton"></img>  

### 🚗 Mobility average velocity prediction
* __Code__  
```python
from function import *
import pandas as pd

window_size, forecast_size = 24*7,24
raw=pd.read_csv('./data/서인천IC-부평IC 평균속도.csv',encoding='CP949').set_index('집계일시').drop('Unnamed: 0',axis=1)
plt.plot(raw)
plt.show()

''' 1. preprocess raw data '''
date, data=split_data(raw,0,index=True) 

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
```
* __Data__

<img src="https://user-images.githubusercontent.com/83653380/231968376-9aeb537f-457b-4c62-a52d-575fc65c0c5f.png" width="50%" height="38%" title="original mobility velocity"></img>  

* __Prediction__  

<img src="https://user-images.githubusercontent.com/83653380/231968631-0eddb9c0-6216-433e-84cc-3ff796791e88.png" width="80%" height="60%" title="mobility velocity prediciton"></img>  
