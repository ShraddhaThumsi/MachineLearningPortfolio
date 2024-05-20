import utils.ReadLiteratureData as read
import utils.EDA as EDA
import utils.CleanPreprocessData as Preprocess
from sklearn.model_selection import train_test_split
import utils.Models as Models
train_data,test_data = read.read_and_clean_df()
print(train_data.shape)
EDA.analyze_gene_data(train_data)
train_data['Text']=train_data['Text'].apply(Preprocess.clean_text)
print(train_data.shape)
print(train_data['Text'])
y_true = train_data['Class'].astype(int)
del train_data['Class']
#data and code reference from ProjectPro
#we want the same distribution on the train,validation and test sets, so we will split accordingly
X_train, X_rem, y_train, y_rem = train_test_split(train_data, y_true, stratify=y_true, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, stratify=y_rem, test_size=0.5)
X_train_gene_onehot,X_val_gene_onehot,X_test_gene_onehot=Models.transform_features_into_numer(X_train['Gene'],X_val['Gene'],X_test['Gene'])
X_train_variation_onehot,X_val_variation_onehot,X_test_variation_onehot=Models.transform_features_into_numer(X_train['Variation'],X_val['Variation'],X_test['Variation'])
X_train_text_feature,X_val_text_feature,X_test_text_feature=Models.extract_text_features(X_train['Text'],X_val['Text'],X_test['Text'],min_df_value=400)
print('showing shape of text feature of training data, validation data and testing data in that order')
print(X_train_text_feature.shape)
print(X_val_text_feature.shape)
print(X_test_text_feature.shape)
X_train_new=Models.concatenate_features(X_train_gene_onehot,X_train_variation_onehot,X_train_text_feature,print_this='train')
X_val_new=Models.concatenate_features(X_val_gene_onehot, X_val_variation_onehot,X_val_text_feature,print_this='val')
X_test_new=Models.concatenate_features(X_test_gene_onehot, X_test_variation_onehot, X_test_text_feature,print_this='test')
print('shape of training data after feature concatenation',X_train_new.shape)
print('shape of validation data after feature concatenation',X_val_new.shape)
print('shape of testing data after feature concatenation',X_test_new.shape)
print('now invoking baseline logistic regression model')
print(Models.logistic_regression(X_train_new, y_train,loop='training'))
print(Models.logistic_regression(X_val_new,y_val,loop='validation'))
print(Models.logistic_regression(X_test_new,y_test,loop='test'))