import numpy as np
import matplotlib.pyplot as plt
def white_noise(df):
    np.random.seed(10)
    white_noise = np.random.normal(loc=df['IOT_Sensor_Reading'].mean(), scale=df['IOT_Sensor_Reading'].std(),size=len(df))
    #df['white noise'] = white_noise
    print(df.describe())
    plt.plot(white_noise)
    plt.xlabel('Time')
    plt.ylabel('Amplitude of noise')
    plt.title('White noise generated from mean and standard deviation from the dataset')
    plt.savefig('./output/white_noise.png')
    return plt.show()
def random_walk(df):
    range_min = df['IOT_Sensor_Reading'].min()
    range_max = df['IOT_Sensor_Reading'].max()
    num_steps=len(df)
    initial_stop = 0
    step_size = np.random.choice([range_min, range_max], size=num_steps)
    random_walk = np.cumsum(step_size)+initial_stop
    #df['random_walk'] = random_walk

    plt.plot(random_walk)
    plt.xlabel('Size of dataset')
    plt.ylabel('step size')
    plt.title('Inducing Random Walk to the dataset')
    plt.savefig('./output/random_walk.png')

    return plt.show()