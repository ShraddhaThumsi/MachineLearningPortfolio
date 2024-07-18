import numpy as np
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
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

def garch_forecast(returns,horizons,is_garch=True,lags=1):
    if is_garch:

        model = arch_model(returns,vol='GARCH',p=lags,q=1)
    else:
        model = arch_model(returns, vol='ARCH',p=lags)
    model_fit=model.fit()
    forecast = model_fit.forecast(horizon=horizons)
    return forecast.variance.values[-1]