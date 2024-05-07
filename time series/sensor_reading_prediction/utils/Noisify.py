import numpy as np
import matplotlib.pyplot as plt
def white_noise(df):
    np.random.seed(10)
    white_noise = np.random.normal(loc=df['IOT_Sensor_Reading'].mean(), scale=df['IOT_Sensor_Reading'].std(),size=len(df))
    df['white noise'] = white_noise
    print(df.describe())
    plt.plot(white_noise)
    plt.xlabel('Time')
    plt.ylabel('Amplitude of noise')
    plt.title('White noise generated from mean and standard deviation from the dataset')
    plt.savefig('./output/white_noise.png')
    return plt.show()