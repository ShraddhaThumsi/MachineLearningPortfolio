import pandas as pd
import matplotlib.pyplot as plt
path_to_file = '../data/linear_regression_part1_soccerdata.csv'
df = pd.read_csv(path_to_file)
print(df.shape)
print(df.describe())
print(df.corr())

plt.scatter(df['Cost'],df['Score'])
plt.show()