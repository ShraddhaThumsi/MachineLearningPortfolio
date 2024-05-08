import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import sys

def MA_model(df):
    order = (0,0,1)
    print('inside ma model function')
    print(df.shape)
    print(df.columns)
    model = ARIMA(df['IOT_Sensor_Reading'], order=order)
    print('inside moving average model function, i have invoked arima successfully')
    model_fit=model.fit()
    prediction = model_fit.fittedvalues
    rmse = np.sqrt(mean_squared_error(df['IOT_Sensor_Reading'], prediction))
    with open('./output/MA_model.txt', 'a') as f:
        sys.stdout = f
        print(f'RMSE of Moving Average model: {rmse}')
        print('summary of Moving Average model')
        print(model_fit.summary())
    sys.stdout = sys.__stdout__

    plt.plot(df['IOT_Sensor_Reading'],label='Original Data')
    plt.plot(prediction,label='Predicted Values')
    plt.legend()
    plt.title('Moving Average Model')
    plt.xlabel('Time in days elapsed')
    plt.ylabel('Value')
    plt.savefig('./output/MA_model.png')
    return plt.show()