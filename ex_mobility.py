from function import *
import pandas as pd
window_size, forecast_size = 24*7,24
raw=pd.read_csv('./data/서인천IC-부평IC 평균속도.csv',encoding='CP949').set_index('집계일시').drop('Unnamed: 0',axis=1)
plt.plot(raw)
plt.show()
date, data=split_data(raw,0,index=True) # preprocess raw data
dataloader=build_dataLoader(data,
                            window_size=window_size,
                            forecast_size=forecast_size,
                            batch_size=4) #make dataloader
pred=trainer(data,
             dataloader,
             window_size=window_size,
             forecast_size=forecast_size).implement() #train and evaluate
figureplot(date,data,pred,window_size,forecast_size) #plot the result.