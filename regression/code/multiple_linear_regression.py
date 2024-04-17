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
