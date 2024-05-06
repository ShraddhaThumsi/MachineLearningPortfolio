import pandas as pd
import sys
import numpy as np
from datetime import datetime
import utils.PreProcessTimeSeriesData as PreProcess
import utils.EDA as EDA
import utils.StationarityTests as Stationary
path_to_input_file = './data/Data-Chillers.csv'
sensor_data_df = pd.read_csv(path_to_input_file)
processed_df = PreProcess.clean_time_series(sensor_data_df)
print(processed_df.columns)
df = processed_df.set_index('time')#the primary index in the original df is a row number- an integer value. When we make the time column as the index, the only other column in the dataset is the sensor value. The covariance matrix is defined for only one feature at a time
print(df.columns)
with open('./output/'+'covariance_log.txt','a') as f:
    sys.stdout=f
    print('Run time: ', datetime.now())
    print('*********Covariance matrix is as follows: *******')
    print(np.cov(df,rowvar=False))
sys.stdout =  sys.__stdout__

components_plot = EDA.timeseries_eda(processed_df)
stationarity_plot = Stationary.check_stationarity_tests(processed_df)