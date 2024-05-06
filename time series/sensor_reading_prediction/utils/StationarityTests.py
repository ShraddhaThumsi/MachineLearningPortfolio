from statsmodels.tsa.stattools import adfuller,kpss
import matplotlib.pyplot as plt
import sys

def check_stationarity_tests(df):

    with open('./output/StationarityTestsLogs.txt', 'w') as f:
        sys.stdout = f
        plt.plot(df['IOT_Sensor_Reading'])
        plt.title('Original Time Series Data')
        plt.show()

        ad_fuller_result = adfuller(df['IOT_Sensor_Reading'])
        print('ADF statistic : ', ad_fuller_result[0])
        print('p value: ',ad_fuller_result[1])
        print('critical values: ')
        for key,value in ad_fuller_result[4].items():
            print(key,' : ',value)

        diff_data = df.diff().dropna() #by examining out adf result we can see that our data is non-stationary. To remedy this, we can perform cleaning by taking a difference of a timestamp with the previous time stamp. This often removes any trends, seasonalities and residue, and pushes the data towards being stationary.
        plt.plot(diff_data)
        plt.title('Differenced Time Series')
        plt.show()

        #now i will call adfuller on my differenced data. If my differenced data turns out to be stationary, then i can handle my original data to make it stationary for further use.
        ad_fuller_result_on_diff = adfuller(diff_data['IOT_Sensor_Reading']) #examining the statistics indicates that we can assume the differenced data is not non-stationary, i.e. the differenced data is more stationary than the original data
        print('ADF statistic on differenced data: ', ad_fuller_result_on_diff[0])
        print('p value: ', ad_fuller_result_on_diff[1])
        print('critical values: ')
        for key, value in ad_fuller_result_on_diff[4].items():
            print(key, ' : ', value)

        print('---------- Now we will use KPSS tests ----------')
        kpss_result = kpss(df['IOT_Sensor_Reading'])
        plt.plot(df['IOT_Sensor_Reading'])
        plt.title('Original Time Series Data')
        plt.show()
        print(' KPSS statistic on original data: ', kpss_result[0])
        print('p value: ', kpss_result[1])
        print('critical values: ')
        for key, value in kpss_result[3].items():
            print(f'{key}: {value}')

        kpss_result_diff = kpss(diff_data['IOT_Sensor_Reading'])
        plt.plot(diff_data['IOT_Sensor_Reading'])
        plt.title('Differenced Time Data')
        plt.show()
        print('KPSS statistic on differenced data: ', kpss_result_diff[0])
        print('p value: ',kpss_result[1])
        print('critical values: ')
        for key, value in kpss_result_diff[3].items():
            print(key, ' : ', value)



    sys.stdout = sys.__stdout__
    return df