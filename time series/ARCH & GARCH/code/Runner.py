import ReadAndProcessStockPriceData as NiftyReader
import PredictNifty50FuturePrices as Predictor
import MarketSimulatorUtils as MktSimUtils
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
    arima_forecast_prices = MktSimUtils.get_arima_forecast(company_wise_data['VWAP'],horizons=horizons)
    print(f'got the arima forecast prices until {horizons} horizons for company {company}')