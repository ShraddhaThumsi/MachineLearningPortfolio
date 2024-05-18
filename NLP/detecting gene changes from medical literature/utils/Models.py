from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn import metrics

def evaluate_model(model,test_features,y_truth):
    prediction=model.predict(test_features)
    metrics.confusion_matrix(y_truth,prediction)
    predict_probabilities= model.predict_proba(test_features)
    logloss_metric=log_loss(y_truth,predict_probabilities)
    return logloss_metric