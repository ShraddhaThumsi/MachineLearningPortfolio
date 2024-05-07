import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
def calculate_acf(df):
    plt.figure(figsize=(12,4))
    plot_acf(df['IOT_Sensor_Reading'],lags=100)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation function (ACF)')
    plt.savefig('./output/'+'ACF.png')
    return plt.show()
def calculate_pacf(df):
    plt.figure(figsize=(12,4))
    plot_pacf(df['IOT_Sensor_Reading'],lags=100)
    plt.xlabel('Lag')
    plt.ylabel('Partial Autocorrelation')
    plt.title('Partial Autocorrelation function (PACF)')
    plt.savefig('./output/'+'PACF.png')
    return plt.show()