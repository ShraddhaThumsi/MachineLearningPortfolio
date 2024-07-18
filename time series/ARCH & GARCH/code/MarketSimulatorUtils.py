import numpy as np
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score

def calculate_future_prices(current_price,future_volatilities,mu):
    future_prices = []
    for next_day_vol in future_volatilities:
        future_price = current_price*np.exp((mu-0.5*next_day_vol)+np.sqrt(next_day_vol)*np.random.randn())
        future_prices.append(future_price)
    return future_prices

def get_arima_forecast(prices,horizons):
    model = ARIMA(prices,order=(1,1,1))
    model_fit = model.fit()
    forecast=model_fit.predict(steps=horizons)
    return forecast

def garch_forecast(returns,horizons,is_garch=True,p=1,q=1):
    if is_garch:

        model = arch_model(returns,vol='GARCH',p=p,q=q)
    else:
        model = arch_model(returns, vol='ARCH',p=p)
    model_fit=model.fit()
    forecast = model_fit.forecast(horizon=horizons)
    return forecast.variance.values[-1]

def evaluate_model(y_true,y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print('RMSE is {}'.format(rmse))
    print('MAE is {}'.format(mae))
