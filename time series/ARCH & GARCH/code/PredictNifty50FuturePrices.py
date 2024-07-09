import matplotlib.pyplot as plt
def split_train_test(df,test_size=0.2):
    test_index = round(test_size*df.shape[0])
    df_train=df[:-test_index]
    df_test=df[-test_index:]
    return df_train,df_test
def calculate_pct_change_in_volatility(df):
    df['returns'] = df['VWAP'].pct_change(1)*100
    df['squared returns'] = df['returns'].mul(df['returns'])
    df['returns'].plot(figsize=(20, 5))
    plt.title("Returns", size=24)
    plt.show()
    df['squared returns'].plot(figsize=(20, 5))
    plt.title("Squared Returns", size=24)
    plt.show()
