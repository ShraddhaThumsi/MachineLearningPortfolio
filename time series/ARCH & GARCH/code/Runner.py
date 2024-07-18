import ReadAndProcessStockPriceData as NiftyReader
import PredictNifty50FuturePrices as Predictor
import MarketSimulatorUtils as MktSimUtils
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from matplotlib.ticker import FuncFormatter
import datetime

source_of_csv = '/Users/shraddha/datascience/python_code/MachineLearningPortfolio/time series/ARCH & GARCH/data/NIFTY 50'
path_to_output = '/Users/shraddha/datascience/python_code/MachineLearningPortfolio/time series/ARCH & GARCH/output'
nifty_timeseries_df_per_company = NiftyReader.make_dataset(source_of_csv,path_to_output)

horizons=100
#list_of_companies_to_use = ['ADANI_PORTS','ADANI_ENTERPRISES','APOLLO HOSPITALS']
#list_of_companies_to_use = ['ADANI_ENTERPRISES']
list_of_companies_to_use = list(nifty_timeseries_df_per_company.keys())
#column_to_use = 'VWAP'
#column_to_use = 'Adj Close'
column_to_use='Close'
mu = 0.05 # annual risk free return rate in India
y_true = []
y_pred_arima=[]
y_pred_arch=[]
y_pred_garch=[]





#the ARCH and GARCH models give us the volatility in the price of the stock. That's interesting, but we can beef it up by predicting the stock price itself, and also simulating market conditions to get a full picture. Upto now we performed a basic GARCH model. Now we will direct the rest of the code into a stock price predictor set-up. We will also attemot to deploy it.
for company in list_of_companies_to_use:
    company_wise_data = nifty_timeseries_df_per_company[company]
    true_prices = company_wise_data[column_to_use][-horizons:]
    y_true.extend(true_prices)

    print(f'for company {company} the following are the true prices of the stock for past {horizons} days')
    print(true_prices)
    arima_forecast_prices = MktSimUtils.get_arima_forecast(company_wise_data[column_to_use],horizons=horizons)
    y_pred_arima.extend(arima_forecast_prices[-horizons:])
    returns = (np.diff(company_wise_data[column_to_use]) / company_wise_data[column_to_use][:-1])
    print(returns)
    garch_volatilities = MktSimUtils.garch_forecast(returns, horizons,p=3,q=3)
    print(f'garch volatilities are : {garch_volatilities}')

    arch_volatilities=MktSimUtils.garch_forecast(returns, horizons,is_garch=False,p=3)
    print(f'arch volatilities are : {arch_volatilities}')

    current_price = company_wise_data[column_to_use][-1]
    last_available_dateas_str = list(company_wise_data.index)[-1]
    last_available_date = date(int(last_available_dateas_str[-4:]),int(last_available_dateas_str[3:5]),int(last_available_dateas_str[0:2]))
    date_list = [last_available_date - datetime.timedelta(days=x) for x in range(horizons)]

    date_list=sorted([str(d) for d in date_list])
    #there is something worth pointing here. The results from a GARCH model are tethered to a date with the most complete information. We have to recallibrate the garch model each day when new stovk prices become official. But since that is out of scope pf this project, I will assume the same volatility across and use that to predict for any day in the future.
    garch_future_prices = MktSimUtils.calculate_future_prices(current_price, garch_volatilities,mu)
    print(f'future prices from garch vols: {garch_future_prices}')
    y_pred_garch.extend(list(garch_future_prices))
    arch_future_prices = MktSimUtils.calculate_future_prices(current_price, arch_volatilities,mu)
    y_pred_arch.extend(list(arch_future_prices))
    print(f'future prices from arch vols: {arch_future_prices}')
    plt.figure(figsize=(20, 8))
    plt.plot(date_list, garch_future_prices, marker='x', linestyle='-', color='red',
             label='Future Prices (GARCH Forecasted Volatility)')
    plt.plot(date_list, arima_forecast_prices[-horizons:], marker='o', linestyle='-', color='green',
             label='Future Prices (ARIMA Forecasted)')
    plt.plot(date_list, arch_future_prices, marker='v', linestyle='-', color='cyan',
             label='Future Prices (ARCH Forecasted)')
    plt.plot(date_list, true_prices, marker='*', linestyle='-', color='black',
             label=f'true prices of {column_to_use} for {horizons} days')
    plt.xticks(rotation=60)
    plt.legend()

    plt.title(f'{column_to_use} forecasts for {company} by ARIMA, ARCH & GARCH on {horizons} horizons')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
    plt.savefig(path_to_output + '/logs/' + f'stock_price_predictions_{horizons}_horizons_{company}' + '.png')
    plt.close()
    plt.show()
    print(
        f'I have added the last {horizons} stock prices from {len(list_of_companies_to_use)} company,now i will show you the sizes of each of the y_pred and y_true')
    print(len(y_true))
    print(len(y_pred_arch))
    print(len(y_pred_arima))
    print(len(y_pred_garch))
    print('arima results')
    MktSimUtils.evaluate_model(y_true, y_pred_arima)
    print('garch results')
    MktSimUtils.evaluate_model(y_true, y_pred_garch)
    print('arch results')
    MktSimUtils.evaluate_model(y_true, y_pred_arch)


    y_true.clear()
    y_pred_arima.clear()
    y_pred_arch.clear()
    y_pred_garch.clear()

