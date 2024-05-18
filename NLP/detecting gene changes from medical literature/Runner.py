import utils.ReadLiteratureData as read
import utils.EDA as EDA
import utils.CleanPreprocessData as Preprocess
from sklearn.model_selection import train_test_split
train_data,test_data = read.read_and_clean_df()
print(train_data.shape)
EDA.analyze_gene_data(train_data)
train_data['Text']=train_data['Text'].apply(Preprocess.clean_text)
print(train_data.shape)
print(train_data['Text'])
y_true = train_data['Class'].astype(int)
del train_data['Class']
#we want the same distribution on the train,validation and test sets, so we will split accordingly
X_train, X_rem, y_train, y_rem = train_test_split(train_data, y_true, stratify=y_true, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, stratify=y_rem, test_size=0.5)