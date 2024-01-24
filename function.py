import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import Dlinear,Nlinear
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

def split_data(data,target,index=False):
    if index==False:
        result=data.loc[:,target]
    else:
        result=data.iloc[:,target]
    return list(result.index), result.to_numpy()

def transform(raw,check_inverse=False):
    data=raw.reshape(-1,1)
    if not check_inverse:
        return scaler.fit_transform(data)
    else:
        return scaler.inverse_transform(data)[:,0]
    
class windowDataset(Dataset):
    def __init__(self, y, input_window, output_window, stride=1):
        #총 데이터의 개수
        L = y.shape[0]
        #stride씩 움직일 때 생기는 총 sample의 개수
        num_samples = (L - input_window - output_window) // stride + 1
        #input과 output
        X,Y = np.zeros([input_window, num_samples]), np.zeros([output_window, num_samples])

        for i in np.arange(num_samples):
            start_x = stride*i
            end_x = start_x + input_window
            X[:,i] = y[start_x:end_x]

            start_y = stride*i + input_window
            end_y = start_y + output_window
            Y[:,i] = y[start_y:end_y]

        X = X.reshape(X.shape[0], X.shape[1], 1).transpose((1,0,2)) #X:(num_samples,input_window,1)
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1).transpose((1,0,2)) #Y:(num_samples,output_window,1)
        self.x, self.y = X, Y
        self.len = len(X)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
    
    def __len__(self):
        return self.len

def build_dataLoader(data,window_size:int,forecast_size:int,batch_size:int):
    train=transform(data)[:-window_size,0]
    dataset=windowDataset(train,window_size,forecast_size)
    result=DataLoader(dataset,batch_size=batch_size)
    return result

class trainer():
    def __init__(self, data, dataloader, window_size, forecast_size, name="DLinear", feature_size=4, lr=1e-4):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.data=data
        self.trains=transform(data)[:-window_size,0]
        self.dataloader=dataloader
        self.window_size=window_size
        self.forecast_size=forecast_size
        if name=="DLinear":
            self.model=Dlinear(window_size,forecast_size).to(self.device)
        elif name=="NLinear":
            self.model=Nlinear(window_size,forecast_size).to(self.device)
        else:
            raise(ValueError("model name이 정확하지 않습니다. DLinear 또는 NLinear로 입력하셨는지 확인해주세요."))
        self.feature_size=feature_size
        self.name=name
        self.criterion = nn.MSELoss()
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=lr)

    def train(self, epoch=65):
        self.model.train()
        progress=tqdm(range(epoch))
        losses=[]
        for _ in progress:
            batchloss = 0.0
            for (inputs, outputs) in self.dataloader:
                self.optimizer.zero_grad()
                result = self.model(inputs.float().to(self.device))
                loss = self.criterion(result, outputs.float().to(self.device))
                loss.backward()
                self.optimizer.step()
                batchloss += loss
            losses.append(batchloss.cpu().item())
            progress.set_description("loss: {:0.6f}".format(batchloss.cpu().item() / len(self.dataloader)))
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('epoch vs loss graph')
        plt.plot(losses)

    def evaluate(self):
        window_size=self.window_size
        input = torch.tensor(self.trains[-window_size:]).reshape(1,-1,1).float().to(self.device)
        self.model.eval()
        predictions = self.model(input)
        return predictions.detach().cpu().numpy()
    
    def implement(self):
        process=trainer(self.data,self.dataloader,self.window_size,
                        self.forecast_size,self.name,self.feature_size)
        process.train()
        evaluate=process.evaluate()
        result=transform(evaluate,check_inverse=True)
        return result

def figureplot(date,data,pred,window_size,forecast_size):
    datenum=mdates.date2num(date)
    len=data.shape[0]
    fig, ax = plt.subplots(figsize=(20,5))
    ax.plot(datenum[len-window_size:len], data[len-window_size:], label="Real")
    ax.plot(datenum[len-forecast_size:len], pred, label="LSTM-linear")
    locator = mdates.AutoDateLocator()
    formatter = mdates.AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.xlabel('date')
    plt.ylabel('values')
    plt.title('Comparison between prediction and actual values')
    plt.legend()
    plt.show()