from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

def timeseries_eda(df):
    df.set_index("time", inplace=True)
    components_of_timeseries = seasonal_decompose(df['IOT_Sensor_Reading'],model='additive',period=365)
    trend = components_of_timeseries.trend
    seasonal = components_of_timeseries.seasonal
    irregular = components_of_timeseries.resid
    # Plot the components
    plt.subplot(411)
    plt.plot(df['IOT_Sensor_Reading'], label='Original')
    plt.legend(loc='upper left')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='upper left')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='upper left')
    plt.subplot(414)
    plt.plot(irregular, label='Residuals')
    plt.legend(loc='upper left')
    plt.tight_layout()
    return plt.show()