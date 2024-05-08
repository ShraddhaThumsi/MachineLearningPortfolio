import pandas as pd
def clean_time_series(time_series_df,is_univariate=True,should_set_index=True):
    time_series_df.time=pd.to_datetime(time_series_df.time, format='%d-%m-%Y %H:%M') #the format here represents the format in which the data appears, not the format in which we want to store the cleaned up date column
    time_series_df=time_series_df.sort_values('time')
    if is_univariate:
        del time_series_df['Error_Present']
        del time_series_df['Sensor_2']
        del time_series_df['Sensor_Value']
    if should_set_index:
        time_series_df.set_index('time',inplace=True)
    time_series_df = time_series_df.asfreq('H')
    time_series_df.IOT_Sensor_Reading = time_series_df.IOT_Sensor_Reading.fillna(method='ffill')
    if is_univariate is False:
        time_series_df.Error_Present = time_series_df.Error_Present.fillna(method='ffill')
        time_series_df.Sensor_2 = time_series_df.Sensor_2.fillna(method='ffill')
        time_series_df.Sensor_Value = time_series_df.Sensor_Value.fillna(method='ffill')
    if should_set_index:
        time_series_df.reset_index(inplace=True)

    return time_series_df