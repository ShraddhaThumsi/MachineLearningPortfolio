import os
import pandas as pd
import matplotlib.pyplot as plt

def make_dataset(path_to_nifty,path_to_output):
    nifty_dir = sorted(os.listdir(path_to_nifty))
    dataset = {}
    unwanted_files = ['NIFTY_50_STOCKS.csv']

    nifty_dir = filter(lambda x : x not in unwanted_files, nifty_dir)
    for nifty in nifty_dir:
        print(nifty)
        #my dataset has many columns, like opening price, closing price, adjusted closing price etc. I will take only the adjusted closing price of the stock for the rest of my work.
        #Forecasts can be gotten for the remaining columns by simply replacing the column we want.
        company_wise_data = pd.read_csv(path_to_nifty+'/'+nifty)
        #we can check which columns have missing values and make sure to interpolate that before further analyses


        #in this dataset, we are dealing with stock price values. The way stock prices are recorded, we will have missing data on weekends, national holidays etc. So for rows with no data, it is kosher to just drop the row, pretending as though prices were not recorded on that day.
        # However, this dropping should be done based on a judgement call. Heart rate monitors may not be applicable to drop missing rows, imputation will be better suited there.
        company_wise_data = company_wise_data.dropna(how='any', axis=0)

        #company_wise_data['Adj Close'] = company_wise_data['Adj Close'].interpolate(option='spline')
        company_wise_data['VWAP'] = company_wise_data['VWAP'].astype(float)
        company_wise_data.set_index('Date',inplace=True)

        dataset[nifty[0:-4]] = company_wise_data
        # plt.figure(figsize=(20,5))
        # plt.plot(company_wise_data['VWAP'],label='VWAP')
        # plt.plot(company_wise_data['Adj Close'],label='Adjusted Close')
        # plt.title(f'{nifty[0:-4]}')
        # plt.legend(loc='upper right')
        # plt.xticks(rotation=45)
        # plt.savefig(path_to_output+'/EDA/'+nifty[0:-4]+'.png')
        # plt.close()

    # plt.show()
    return dataset
