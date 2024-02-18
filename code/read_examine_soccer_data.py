import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
path_to_file = '../data/linear_regression_part1_soccerdata.csv'
df = pd.read_csv(path_to_file)



#EDA on our soccer dataset
#print(df['Club'].value_counts())

#Statistical features of each column of dataset, including quartiles and standard deviation
#print(df.describe())

#uncovering the datatype of each column
#print(df.info())

#examining the pearson correlation matrix
#print(df.corr())
sns.heatmap(df.corr())
plt.show()

#cost to hire the player in a club and the score obtained by them in a match is positively correlated with a Pearson coefficient of +0.96
plt.scatter(df['Cost'],df['Score'])
plt.title('Cost to hire one player v/s score obtained by them')
#plt.show()

#the distance run by a player in a game and the score obtained by them is negatively correlated with a Pearson coefficient of -0.49.
#This means that longer the distance that a player runs befor ehe makes a goal, the lower his game score will be.


plt.scatter(df['DistanceCovered(InKms)'],df['Score'])
plt.title('Distance covered v/s score obtained by one player')
#plt.show()


# Let's examine this with the time taken by the player to make a goal -
plt.scatter(df['MinutestoGoalRatio'],df['Score'])
plt.title('Time taken to make a goal v/s score obtained by one player')
#plt.show()
#as expected, the longer the player takes before making his goal, the lower his game score will be