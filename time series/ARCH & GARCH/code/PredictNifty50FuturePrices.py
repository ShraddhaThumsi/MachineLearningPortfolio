import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
from arch import arch_model
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


def make_arch_model(df,path_to_output):
    #combination 1 - Default parameters
    file=open(path_to_output+'/logs/' + 'arch_model_default.txt','w')
    model_arch_1 = arch_model(df['returns'][1:])
    results_arch1 = model_arch_1.fit(update_freq=5)
    file.write(str(results_arch1.summary()))
    file.close()
    print(results_arch1.summary())

    file = open(path_to_output+'/logs/' + 'arch_model_meanconst_volarch_p1_distnorm.txt','w')
    model_arch_1 = arch_model(df['returns'][1:],mean='Constant',vol='ARCH',p=1,dist='normal')
    results_arch1 = model_arch_1.fit(update_freq=5)
    file.write(str(results_arch1.summary()))
    file.close()
    print(results_arch1.summary())

    file = open(path_to_output + '/logs/' + 'arch_model_meanconst_volarch_p4_distnorm.txt', 'w')
    model_arch_1 = arch_model(df['returns'][1:], mean='Constant', vol='ARCH', p=4, dist='normal')
    results_arch1 = model_arch_1.fit(update_freq=5)
    file.write(str(results_arch1.summary()))
    file.close()
    print(results_arch1.summary())

    #the best model will be returned. By increasing the p value which is number of lags, we see that past 4th lag the results are not statistically significant, therefore we will stop at p=4.
    return results_arch1

def make_garch_model(df, path_to_output):
    file=open(path_to_output+'/logs/'+'garch_1_1.txt','w')
    model_garch_1_1 = arch_model(df['returns'][1:],mean='Constant',vol='GARCH',p=1,q=1,dist='normal')
    results_garch_1_1 =model_garch_1_1.fit(update_freq=5)
    file.write(str(results_garch_1_1))
    file.close()
    print(results_garch_1_1.summary())


    return results_garch_1_1