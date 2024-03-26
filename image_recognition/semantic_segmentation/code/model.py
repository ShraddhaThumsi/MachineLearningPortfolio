#reference https://medium.com/@abdualimov/unet-implementation-of-the-unet-architecture-on-tensorflow-for-segmentation-of-cell-nuclei-528b5b6e6ffd
from keras import Input
import tensorflow as tf
import task
import data_preparation
import utils
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau




X_data= data_preparation.X_data
Y_data= data_preparation.Y_data
X_train, X_test, Y_train, Y_test = task.train_test_split(X_data, Y_data, test_size=0.1,random_state=13)
train_generator,val_generator,test_generator = utils.augment_data(X_train, Y_train,X_test, Y_test,X_data)
print(type(train_generator))
print(type(val_generator))
print(type(test_generator))

std_img_width = task.IMG_WIDTH
std_img_height = task.IMG_HEIGHT
num_channels = task.IMG_CHANNELS
def make_model(input_shape=(std_img_width,std_img_height,num_channels),filter_list=[32,64,128,256,512]):
    tf.keras.backend.clear_session()
    inputs = Input(shape=input_shape)
    s= (lambda x: x/255.0)(inputs)

    c1=Conv2D(32,(3,3),activation='elu',kernel_initializer='he_normal',padding='same')(s)
    c1 = Dropout(0.5)(c1)
    c1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    c1 = Dropout(0.5)(c1)
    p1 = MaxPooling2D((2,2),strides=(2,2))(c1)

    c2 = Conv2D(64,(3,3),activation='elu',kernel_initializer='he_normal',padding='same')(p1)
    c2 = Dropout(0.5)(c2)
    c2 = Conv2D(64,(3,3),activation='elu',kernel_initializer='he_normal',padding='same')(c2)
    c2 = Dropout(0.5)(c2)
    p2 = MaxPooling2D((2,2),strides=(2,2))(c2)

    up1_2 = Conv2DTranspose(filter_list[0], (2,2), strides=(2,2), name='up12', padding='same')(c2)
    conv1_2 = concatenate([up1_2, c1], name='merge12', axis=3)
    c3 = Conv2D(32, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(conv1_2)
    c3 = Dropout(0.5)(c3)
    c3 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    c3 = Dropout(0.5)(c3)

    conv3_1 = Conv2D(128, (3,3), activation='elu', kernel_initializer='he_normalizer', padding='same')(p2)
    conv3_1 = Dropout(0.5)(conv3_1)
    conv3_1 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normalizer', padding='same')(conv3_1)
    conv3_1 = Dropout(0.5)(conv3_1)
    p3 = MaxPooling2D((2,2),(2,2),name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(filter_list[1], (2,2), strides=(2,2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2,c2],name='merge22',axis=3)
    conv2_2 = Conv2D(64,(3,3),activation='elu',kernel_initializer='he_normal',padding='same')(conv2_2)
    conv2_2 = Dropout(0.5)(conv2_2)
    conv2_2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv2_2)
    conv2_2 = Dropout(0.5)(conv2_2)

    up1_3 = Conv2DTranspose(filter_list[0],(2,2),(2,2),name='up13',padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3,c1,c3],name='merge13',axis=3)
    conv1_3 = Conv2D(32,(3,3), activation='elu', kernel_initializer='he_normal', padding='same')(conv1_3)
    conv1_3 = Dropout(0.5)(conv1_3)
    conv1_3 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv1_3)
    conv1_3 = Dropout(0.5)(conv1_3)

    conv4_1 = Conv2D(256,(3,3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    conv4_1 = Dropout(0.5)(conv4_1)
    conv4_1 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv4_1)
    conv4_1 = Dropout(0.5)(conv4_1)
    p4 = MaxPooling2D((2,2),(2,2),name='pool4')(conv4_1)









    return s
print(make_model().shape)




