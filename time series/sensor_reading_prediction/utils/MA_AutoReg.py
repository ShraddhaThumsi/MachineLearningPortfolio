import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import sys
from statsmodels.tsa.ar_model import AutoReg

def MA_model(df):
    order = (0,0,1)

    model = ARIMA(df['IOT_Sensor_Reading'], order=order)

    model_fit=model.fit()
    prediction = model_fit.fittedvalues
    rmse = np.sqrt(mean_squared_error(df['IOT_Sensor_Reading'], prediction))
    with open('./output/MA_model.txt', 'w') as f:
        sys.stdout = f
        print(f'RMSE of Moving Average model: {rmse}')
        print('summary of Moving Average model')
        print(model_fit.summary())
    sys.stdout = sys.__stdout__

    # plt.plot(df['IOT_Sensor_Reading'],label='Original Data')
    # plt.plot(prediction,label='Predicted Values')
    # plt.legend()
    # plt.title('Moving Average Model')
    # plt.xlabel('Time in days elapsed')
    # plt.ylabel('Value')
    # plt.savefig('./output/MA_model.png')
    return plt.show()


def AR_model(df):

    # Create AR model
    order = 1  # Order of the AR model
    model = AutoReg(df['IOT_Sensor_Reading'], lags=order)

    # Fit the model
    model_fit = model.fit()

    # Get the fitted values
    fitted_values = model_fit.fittedvalues

    rmse = np.sqrt(mean_squared_error(df['IOT_Sensor_Reading'][:-1], fitted_values))
    with open('./output/' + 'log.txt', 'a') as f:
        sys.stdout = f
        print("***************************AR1 Model*******************************")
        print(f"RMSE of AR1 model: {rmse}")
        print(model_fit.summary())
    sys.stdout = sys.__stdout__

    # Plot the original data and the fitted values
    plt.plot(df['IOT_Sensor_Reading'], label='Original Data')
    plt.plot(fitted_values, label='Fitted Values')
    plt.legend()
    plt.title('First-Order AR Model')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.savefig('./output/' + 'AR_Model.png')
    plt.show()


    return plt.show()