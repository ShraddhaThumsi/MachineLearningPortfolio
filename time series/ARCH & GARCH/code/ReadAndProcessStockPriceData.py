import os
import pandas as pd
source_of_csv = '/Users/shraddha/datascience/python_code/MachineLearningPortfolio/time series/ARCH & GARCH/data/NIFTY 50'
print((os.listdir(source_of_csv))[0])
def make_dataset(path_to_nifty):
    nifty_dir = os.listdir(path_to_nifty)
    dataset = {}
    unwanted_files = ['NIFTY_50_STOCKS.csv']

    nifty_dir = filter(lambda x : x not in unwanted_files, nifty_dir)
    for nifty in nifty_dir:
        print(nifty)
        company_wise_data = pd.read_csv(source_of_csv+'/'+nifty)
        dataset[nifty[0:-4]] = company_wise_data
        print(company_wise_data.shape)


    return dataset
make_dataset(source_of_csv)