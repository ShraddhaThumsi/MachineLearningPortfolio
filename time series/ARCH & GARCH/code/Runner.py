import ReadAndProcessStockPriceData as NiftyReader
import PredictNifty50FuturePrices as Predictor
import MarketSimulatorUtils as MktSimUtils
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import datetime
source_of_csv = '/Users/shraddha/datascience/python_code/MachineLearningPortfolio/time series/ARCH & GARCH/data/NIFTY 50'
path_to_output = '/Users/shraddha/datascience/python_code/MachineLearningPortfolio/time series/ARCH & GARCH/output'
nifty_timeseries_df_per_company = NiftyReader.make_dataset(source_of_csv,path_to_output)
print(type(nifty_timeseries_df_per_company))
horizons=20
list_of_companies_to_use = ['ADANI_ENTERPRISES']
#list_of_companies_to_use = list(nifty_timeseries_df_per_company.keys())
for company in list_of_companies_to_use:
    print(f'printing {horizons} horizons for company {company}')
    train,test=Predictor.split_train_test(nifty_timeseries_df_per_company[company])
    Predictor.calculate_pct_change_in_volatility(train,path_to_output)
    trained_arch1_model = Predictor.make_arch_model(train,path_to_output)
    trained_garch_model = Predictor.make_garch_model(train,path_to_output)
    arch_prediction = trained_arch1_model.forecast(horizon=horizons)
    print(arch_prediction.residual_variance)
    garch_prediction = trained_garch_model.forecast(horizon=horizons)
    print(garch_prediction.residual_variance)


#the ARCH and GARCH models give us the volatility in the price of the stock. That's interesting, but we can beef it up by predicting the stock price itself, and also simulating market conditions to get a full picture. Upto now we performed a basic GARCH model. Now we will direct the rest of the code into a stock price predictor set-up. We will also attemot to deploy it.
for company in list_of_companies_to_use:
    company_wise_data = nifty_timeseries_df_per_company[company]
    print(len(list(company_wise_data.index)))
    arima_forecast_prices = MktSimUtils.get_arima_forecast(company_wise_data['VWAP'],horizons=horizons)
    print(f'got the arima forecast prices until {horizons} horizons for company {company}')
    returns = np.diff(company_wise_data['VWAP']) / company_wise_data['VWAP'][:-1]
    volatilities = MktSimUtils.garch_forecast(returns,horizons)
    current_price = company_wise_data['VWAP'][-1]
    last_available_dateas_str = list(company_wise_data.index)[-1]
    last_available_date = date(int(last_available_dateas_str[-4:]),int(last_available_dateas_str[3:5]),int(last_available_dateas_str[0:2]))
    print(last_available_date)
    date_list = [last_available_date - datetime.timedelta(days=x) for x in range(horizons)]
    print(len(date_list))
    date_list=[str(d) for d in date_list]
    #there is something worth pointing here. The results from a GARCH model are tethered to a date with the most complete information. We have to recallibrate the garch model each day when new stovk prices become official. But since that is out of scope pf this project, I will assume the same volatility across and use that to predict for any day in the future.
    future_prices = MktSimUtils.calculate_future_prices(current_price,volatilities)
    plt.plot(date_list, future_prices, marker='x', linestyle='-', color='blue',
             label='Future Prices (GARCH Forecasted Volatility)')
    plt.savefig(path_to_output + '/logs/' + f'garch_predictions_of_volatility_{company}' + '.png')
    plt.close()
    plt.show()