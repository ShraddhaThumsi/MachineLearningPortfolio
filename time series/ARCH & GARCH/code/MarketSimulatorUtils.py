import numpy as np
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
def calculate_future_prices(current_price,future_volatilities):
    future_prices = []
    for next_day_vol in future_volatilities:
        future_price = current_price*np.exp(np.sqrt(next_day_vol))
        future_prices.append(future_price)
    return future_prices

def get_arima_forecast(prices,horizons):
    model = ARIMA(prices,order=(1,1,1))
    model_fit = model.fit()
    forecast=model_fit.predict(steps=horizons)
    return forecast

def garch_forecast(returns,horizons):
    model = arch_model(returns,vol='GARCH',p=1,q=1)
    model_fit=model.fit()
    forecast = model_fit.forecast(horizon=horizons)
    return forecast.variance.values[-1]