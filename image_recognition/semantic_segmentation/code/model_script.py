#reference https://medium.com/@abdualimov/unet-implementation-of-the-unet-architecture-on-tensorflow-for-segmentation-of-cell-nuclei-528b5b6e6ffd
#breast cancer data source https://www.kaggle.com/datasets/andrewmvd/breast-cancer-cell-segmentation
#hippocampus data source - Medical segmentation decathlon http://medicaldecathlon.com
#main inspiration for this project was ProjectPro Image Segmentation module. They had written it in Pytorch so I used existing Tensorflow equivivalents to compose my code.
from sklearn.model_selection import train_test_split

import data_preparation
import utils
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from skimage.io import imshow
import matplotlib.pyplot as plt
import random

from keras.models import load_model
X_data= data_preparation.X_data
Y_data= data_preparation.Y_data
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.1,random_state=13)
train_generator,val_generator,test_generator = utils.augment_data(X_train, Y_train,X_test, Y_test,X_data)


std_img_width = 256
std_img_height = 256
num_channels = 3

LEARNING_RATE = 0.001
NUM_EPOCHS = 3

def make_model(input_shape=(std_img_width,std_img_height,num_channels),filter_list=[32,64,128,256,512]):
    print('inside make model function')
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

    conv3_1 = Conv2D(128, (3,3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    conv3_1 = Dropout(0.5)(conv3_1)
    conv3_1 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv3_1)
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

    up3_2 = Conv2DTranspose(filter_list[2], (2,2),(2,2),name='up32',padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2,conv3_1], axis=3, name='merge32')
    conv3_2 = Conv2D(128,(3,3),activation='elu',kernel_initializer='he_normal',padding='same')(conv3_2)
    conv3_2 = Dropout(0.5)(conv3_2)
    conv3_2 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv3_2)
    conv3_2 = Dropout(0.5)(conv3_2)

    up2_3 = Conv2DTranspose(filter_list[1], (2,2),(2,2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3,c2,conv2_2],name='merge23',axis=3)
    conv2_3 = Conv2D(64,(3,3),activation='elu',kernel_initializer='he_normal',padding='same')(conv2_3)
    conv2_3 = Dropout(0.5)(conv2_3)
    conv2_3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv2_3)
    conv2_3 = Dropout(0.5)(conv2_3)

    up1_4 = Conv2DTranspose(filter_list[0],(2,2),(2,2),name='up14',padding='same')(conv2_3)

    print('created the up14 layer, now need to concatenate with older layers')

    conv1_4 = concatenate([up1_4,c1,c3,conv1_3],name='merge14',axis=3)
    print('concatenation layer conv14 is done')
    conv1_4 = Conv2D(32,(3,3),activation='elu',kernel_initializer='he_normal',padding='same')(conv1_4)
    conv1_4 = Dropout(0.5)(conv1_4)
    conv1_4 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv1_4)
    conv1_4 = Dropout(0.5)(conv1_4)

    conv5_1 = Conv2D(512,(3,3),activation='elu',kernel_initializer='he_normal',padding='same')(p4)
    conv5_1 = Dropout(0.5)(conv5_1)
    conv5_1 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv5_1)
    conv5_1 = Dropout(0.5)(conv5_1)

    up4_2 = Conv2DTranspose(filter_list[3],(2,2),(2,2),name='up42',padding='same')(conv5_1)
    conv4_2= concatenate([up4_2,conv4_1],name='merge42',axis=3)
    conv4_2 = Conv2D(256,(3,3),activation='elu',kernel_initializer='he_normal',padding='same')(conv4_2)
    conv4_2 = Dropout(0.5)(conv4_2)
    conv4_2 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv4_2)
    conv4_2 = Dropout(0.5)(conv4_2)

    up3_3 = Conv2DTranspose(filter_list[2],(2,2),(2,2),name='up33',padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3,conv3_1,conv3_2],name='merge33',axis=3)
    conv3_3 = Conv2D(128,(3,3),activation='elu',kernel_initializer='he_normal',padding='same')(conv3_3)
    conv3_3 = Dropout(0.5)(conv3_3)
    conv3_3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv3_3)
    conv3_3 = Dropout(0.5)(conv3_3)

    up2_4 = Conv2DTranspose(filter_list[1],(2,2),(2,2),name='up24',padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4,c2,conv2_2,conv2_3],name='merge24',axis=3)
    conv2_4 = Conv2D(64,(3,3),activation='elu',kernel_initializer='he_normal',padding='same')(conv2_4)
    conv2_4 = Dropout(0.5)(conv2_4)
    conv2_4 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv2_4)
    conv2_4 = Dropout(0.5)(conv2_4)

    up1_5 = Conv2DTranspose(filter_list[0],(2,2),(2,2),name='up15',padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5,c1,c3,conv1_3,conv1_4],name='merge15',axis=3)
    conv1_5 = Conv2D(32,(3,3),activation='elu',kernel_initializer='he_normal',padding='same')(conv1_5)
    conv1_5 = Dropout(0.5)(conv1_5)
    conv1_5 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv1_5)
    conv1_5 = Dropout(0.5)(conv1_5)

    nesnet_output = Conv2D(1,(1,1),activation='sigmoid',kernel_initializer='he_normal',name='output4',padding='same')(conv1_5)

    model = Model([inputs],[nesnet_output])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),loss='categorical_crossentropy',metrics=['accuracy','mse'])

    return model
#model = make_model()
checkpoint = ModelCheckpoint('best_model.hdf5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             save_weights_only=False,
                             save_freq='epoch')
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.3,
                              patience=5,
                              min_lr=0.0005)
callback_list = [checkpoint, reduce_lr]
# history = model.fit(train_generator,
#                     validation_data=val_generator,
#                     steps_per_epoch=len(X_train) / 7,
#                     validation_steps=10,
#                     callbacks=callback_list,
#                     epochs=NUM_EPOCHS,
#                     verbose=1, )


