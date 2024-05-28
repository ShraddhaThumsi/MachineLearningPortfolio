from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn import metrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def evaluate_model(model,test_features,y_truth):
    prediction=model.predict(test_features)
    sns.heatmap(metrics.confusion_matrix(y_truth,prediction),annot=True)
    predict_probabilities= model.predict_proba(test_features)
    logloss_metric=log_loss(y_truth,predict_probabilities)
    #f1_score_per_classifier = metrics.f1_score(y_truth,prediction)
    #print(f'the f1 score for the present classifier is: {f1_score_per_classifier}')
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

    gene_variation_feature = pd.concat([variation_feature, gene_feature], axis=1)
    text_feature = pd.DataFrame(text_feature.toarray())
    gene_variation_feature.reset_index(drop=True, inplace=True)
    gene_variation_text_feature = pd.concat([text_feature, gene_variation_feature], axis=1)
    return gene_variation_text_feature
def logistic_regression(X_data,y_data,random_state=0,loop='training'):
    model = LogisticRegression(random_state=random_state)
    print('ABOUT TO FIT LOGISTIC REGRESSION MODEL')
    trained_log_model=model.fit(X_data,y_data)
    print('FINISHED FITTING LOGISTIC REGRESSION')
    training_error = evaluate_model(trained_log_model,X_data,y_data)
    print(f'currently in {loop} loop')
    print(f'log loss in this loop is : {training_error}')
    return trained_log_model

def random_forest(X_data,y_data,random_state=0,max_depth=2,loop='training'):

    random_forest_model=RandomForestClassifier(max_depth=max_depth,random_state=random_state)
    random_forest_model.fit(X_data, y_data)
    error = evaluate_model(random_forest_model,X_data,y_data)
    print(f'the {loop} loop is in progress in the random forest and the error obtained is {error}')
    return random_forest_model

def naive_bayes(X_data,y_data,loop='training'):
    gaussian_naivebayes_model = GaussianNB()
    gaussian_naivebayes_model.fit(X_data,y_data)
    error = evaluate_model(gaussian_naivebayes_model,X_data,y_data)
    print(f'the {loop} loop is in progress in the naive bayes classifier, and the error obtained is {error}')
    return gaussian_naivebayes_model

def knn(X_data,y_data,loop='training'):
    knn_model = KNeighborsClassifier(n_neighbors=9)
    knn_model.fit(X_data,y_data)
    error=evaluate_model(knn_model,X_data,y_data)
    print(f'the {loop} loop is in progress in KNN and the error obtained is {error}')
    return knn_model

def support_vector_classifier(X_data,y_data,loop='training'):
    support_vector_classifier_model = SVC(decision_function_shape='ovo',probability=True)
    support_vector_classifier_model.fit(X_data,y_data)
    error=evaluate_model(support_vector_classifier_model,X_data,y_data)
    print(f'the {loop} loop is in progress in the support vector classifier, and the error obtained is {error}')
    return support_vector_classifier_model


