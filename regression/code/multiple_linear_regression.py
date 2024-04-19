import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import pandas as pd
path_to_file = '../data/EPL_Soccer_MLR_LR.csv'
df = pd.read_csv(path_to_file)
correlation_data = df.corr()
print(df.shape)
plt.figure(figsize=(8,6),dpi=100)
heat = sns.heatmap(correlation_data, vmin=-1,vmax=1,center=0,cmap=sns.diverging_palette(20,220,n=200),square=True,annot=True)
heat.set_xticklabels(heat.get_xticklabels(), rotation=45,horizontalalignment='right')
plt.show()

def make_sub_df_basedon_chosencolumns(chosen_columns,given_df=df):
       return given_df[chosen_columns]
# extracting the predictor variables by ignoring the categorical variables

X = make_sub_df_basedon_chosencolumns(['DistanceCovered(InKms)', 'Goals',
       'ShotsPerGame', 'AgentCharges', 'BMI', 'Cost',
       'PreviousClubCost'])
y=df['Score']

x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.75,test_size=0.25,random_state=100)

x_train_intercept = sm.add_constant(x_train)
lr = sm.OLS(y_train,x_train_intercept).fit()
#print(lr.summary())

#The condition number is quite high,meaning the change in one variable affects the  prediction by a lot, because there is a collinear variable that also changes significantly and in turn affects the predictor again.
#we will pick out only a handful of features, hopefully those which are not collinear with each other, so that their effect on the predictor may be isolated

X= make_sub_df_basedon_chosencolumns(['DistanceCovered(InKms)'
        , 'BMI', 'Cost',
       'PreviousClubCost'])
x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.75,test_size=0.25,random_state=100)

x_train_intercept = sm.add_constant(x_train)
lr = sm.OLS(y_train,x_train_intercept).fit()
print(lr.summary())

clubs = set(df.Club)
nominal_features = pd.get_dummies(df['Club'])
df_encoded = pd.concat([df,nominal_features],axis=1)
print(df_encoded.head())

#running the regression with one-hot encoding for club data
X = make_sub_df_basedon_chosencolumns(['DistanceCovered(InKms)', 'BMI', 'Cost','PreviousClubCost','CHE','MUN','LIV'],df_encoded)

x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.75,test_size=0.25,random_state=100)

x_train_intercept = sm.add_constant(x_train)
lr = sm.OLS(y_train,x_train_intercept).fit()
