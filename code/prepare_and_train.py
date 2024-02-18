import read_examine_soccer_data as data_source
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
soccer_df=data_source.df
print(soccer_df.shape)
#for the purpose of this mini-project, let us consider the cost of the player as the predictor variable and score obtained as target variable
#In the "read_examine_soccer_data" file we saw a Pearson coeeficient of +0.96, and hence this choice of predictor variable.

x=np.array(soccer_df['Cost']).reshape(-1,1)
y=np.array(soccer_df['Score']).reshape(-1,1)
#now we will split our available data into train test sets
X_train, X_test, y_train, y_test = train_test_split(x,y,train_size=0.75,test_size=0.25,random_state=50)

#the theoretical material gotten from ProjectPro implements the regression using Statsmodels package. Here I am doing it using sklearn
lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.score(X_test,y_test)) #here score means coefficient of determination
#the statsmodel version has a score of 0.94, and we are at 0.93.