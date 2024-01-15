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