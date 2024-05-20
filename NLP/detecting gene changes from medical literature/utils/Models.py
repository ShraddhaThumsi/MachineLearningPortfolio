from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn import metrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model,test_features,y_truth):
    prediction=model.predict(test_features)
    sns.heatmap(metrics.confusion_matrix(y_truth,prediction),annot=True)
    predict_probabilities= model.predict_proba(test_features)
    logloss_metric=log_loss(y_truth,predict_probabilities)
    plt.show()
    return logloss_metric

def transform_features_into_numer(X_train,X_val,X_test):
    train_feature_onehotCoding = pd.get_dummies(X_train, drop_first=True)

    val_feature_onehotCoding = pd.get_dummies(X_val, drop_first=True)
    # we use reindex to handle the unknown categories which didnt appear in the training data
    val_feature_onehotCoding = val_feature_onehotCoding.reindex(columns=train_feature_onehotCoding.columns,
                                                                fill_value=0)

    test_feature_onehotCoding = pd.get_dummies(X_test, drop_first=True)
    test_feature_onehotCoding = test_feature_onehotCoding.reindex(columns=train_feature_onehotCoding.columns,
                                                                  fill_value=0)

    return train_feature_onehotCoding, val_feature_onehotCoding, test_feature_onehotCoding


def extract_text_features(X_train,X_val,X_test,min_df_value):
    text_vectorizer = TfidfVectorizer(min_df=min_df_value, stop_words="english")

    train_text_feature_onehotCoding = text_vectorizer.fit_transform(X_train)
    val_text_feature_onehotCoding = text_vectorizer.transform(X_val)
    test_text_feature_onehotCoding = text_vectorizer.transform(X_test)

    train_text_feature_onehotCoding = normalize(train_text_feature_onehotCoding, axis=0)
    val_text_feature_onehotCoding = normalize(val_text_feature_onehotCoding, axis=0)
    test_text_feature_onehotCoding = normalize(test_text_feature_onehotCoding, axis=0)

    return train_text_feature_onehotCoding, val_text_feature_onehotCoding, test_text_feature_onehotCoding
def concatenate_features(gene_feature,variation_feature,text_feature,print_this='what is your name?'):
    print()
    print('***********************')
    print('inside concatenate function, checking shape of each incoming feature')
    print(print_this.upper())
    print('gene feature shape', gene_feature.shape)
    print('variation feature shape', variation_feature.shape)
    print('text feature shape', text_feature.shape)
    print('***********************')
    print()

    '''Concatenate all extracted features together'''
    gene_variation_feature=pd.concat([variation_feature,gene_feature],axis=1)
    text_feature= pd.DataFrame(text_feature.toarray())
    gene_variation_feature.reset_index(drop=True, inplace=True)
    gene_variation_text_feature=pd.concat([text_feature,gene_variation_feature],axis=1)
    return gene_variation_text_feature
def logistic_regression(X_data,y_data,random_state=0,loop='training'):
    model = LogisticRegression(random_state=random_state)
    trained_log_model=model.fit(X_data,y_data)
    training_error = evaluate_model(trained_log_model,X_data,y_data)
    print(f'currently in {loop} loop')
    print(f'log loss in this loop is : {training_error}')
    return trained_log_model

