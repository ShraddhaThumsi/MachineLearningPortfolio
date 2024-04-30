import pandas as pd
import utils.PreProcessTimeSeriesData as PreProcess
path_to_input_file = './data/Data-Chillers.csv'
sensor_data_df = pd.read_csv(path_to_input_file)
processed_df = PreProcess.clean_time_series(sensor_data_df)
print(processed_df.shape)
print("first five entries after cleaning")
print(processed_df.head())
print("last five entries after cleaning")
print(processed_df.tail())
