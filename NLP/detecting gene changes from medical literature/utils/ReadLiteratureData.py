import pandas as pd
def read_and_clean_df():
    training_text = pd.read_csv('./data/training_text.csv', engine='python', sep='\|\|', skiprows=1, names=['ID', 'Text'])
    testing_text = pd.read_csv('./data/test_text.csv', engine='python', sep='\|\|', skiprows=1, names=['ID', 'Text'])
    training_variant = pd.read_csv('./data/training_variants.csv')
    testing_variant = pd.read_csv('./data/test_variants.csv')

    train_data = pd.merge(training_text,training_variant,on='ID',how='left')
    test_data = pd.merge(testing_text,testing_variant,on='ID',how='left')
    print(train_data['Text'].isnull().values.any())
    train_data.loc[train_data['Text'].isnull(),'Text'] = train_data['Gene']+train_data['Variation']
    print(train_data['Text'].isnull().values.any())

    print(test_data['Text'].isnull().values.any())
    test_data.loc[test_data['Text'].isnull(),'Text'] = test_data['Gene']+test_data['Variation']
    print(test_data['Text'].isnull().values.any())
    test_data.dropna(inplace=True)
    return train_data,test_data