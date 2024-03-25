#importing necessary libraries
import pandas as pd
import numpy as np
import zipfile
import os
import glob
import random
import sys


import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split as train_test_split
from tensorflow.keras.losses import binary_crossentropy as binary_cross_entropy

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

LEARNING_RATE = 0.001
NUM_EPOCHS=25
EXTENSION = 'png'

