import ReadAndProcessStockPriceData as NiftyReader
import PredictNifty50FuturePrices as Predictor
source_of_csv = '/Users/shraddha/datascience/python_code/MachineLearningPortfolio/time series/ARCH & GARCH/data/NIFTY 50'
path_to_output = '/Users/shraddha/datascience/python_code/MachineLearningPortfolio/time series/ARCH & GARCH/output'
nifty_timeseries_df_per_company = NiftyReader.make_dataset(source_of_csv,path_to_output)
print(type(nifty_timeseries_df_per_company))
train,test=Predictor.split_train_test(nifty_timeseries_df_per_company['TATA CONSUMER PRODUCTS'])
Predictor.calculate_pct_change_in_volatility(train,path_to_output)
trained_arch1_model = Predictor.make_arch_model(train,path_to_output)
trained_garch_model = Predictor.make_garch_model(train,path_to_output)