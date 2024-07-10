import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
def split_train_test(df,test_size=0.2):
    test_index = round(test_size*df.shape[0])
    df_train=df[:-test_index]
    df_test=df[-test_index:]
    return df_train,df_test
def calculate_pct_change_in_volatility(df,path_to_output):
    df['returns'] = df['VWAP'].pct_change(1)*100
    df['squared returns'] = df['returns'].mul(df['returns'])
    df['returns'].plot(figsize=(20, 5))
    plt.title("Returns", size=24)
    plt.xticks()
    plt.savefig(path_to_output+'/logs/'+'returns'+'.png')
    plt.close()
    df['squared returns'].plot(figsize=(20, 5))
    plt.title("Squared Returns", size=24)
    plt.xticks()
    plt.savefig(path_to_output+'/logs/'+'squared_returns'+'.png')
    plt.close()
    plt.show()
    #from the PACF of the returns graph we can conclude that lag 1 has the most significance, with lag 2 also being reasonable. Any more lags after that have no significance on the future prediction.
    sgt.plot_pacf(df['returns'][1:],lags=40,alpha=0.05,zero=False,method='ols')
    plt.title('PACF of returns')
    plt.savefig(path_to_output + '/logs/' + 'returns_pacf' + '.png')
    plt.close()
    plt.show()

    # However, in the square of thr returns, lags 1,2 and 3 have significance whereas it trails off lag 4 onwards.
    sgt.plot_pacf(df['squared returns'][1:], lags=40, alpha=0.05, zero=False, method='ols')
    plt.title('PACF of squared returns')
    plt.savefig(path_to_output + '/logs/' + 'squared_returns_pacf' + '.png')
    plt.close()
    plt.show()


