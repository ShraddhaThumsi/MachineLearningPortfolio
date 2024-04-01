
from keras_preprocessing.image import ImageDataGenerator
smooth = 1e-5

def augment_data(X_train,Y_train,X_test,Y_test,X_data):
    print("inside augment_data")
    data_gen_args = dict(rotation_range=45.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='reflect')


    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)
    test_datagen = ImageDataGenerator()
    X_datagen_val = ImageDataGenerator()
    Y_datagen_val = ImageDataGenerator()
    print(X_train.shape)
    print(Y_train.shape)
    X_datagen.fit(X_train, augment=True, seed=13)
    print('successfully fitted x datagen')
    Y_datagen.fit(Y_train, augment=True, seed=13)
    print('successfully fitted y datagen')
    test_datagen.fit(X_data, augment=True, seed=13)
    X_datagen_val.fit(X_test, augment=True, seed=13)
    Y_datagen_val.fit(Y_test, augment=True, seed=13)
    X_train_augmented = X_datagen.flow(X_train,  batch_size=15, shuffle=True, seed=13)
    Y_train_augmented = Y_datagen.flow(Y_train,  batch_size=15, shuffle=True, seed=13)
    test_augmented = test_datagen.flow(X_data, shuffle=False, seed=13)
    X_train_augmented_val = X_datagen_val.flow(X_test,  batch_size=15, shuffle=True, seed=13)
    Y_train_augmented_val = Y_datagen_val.flow(Y_test,  batch_size=15, shuffle=True, seed=13)

    train_generator = zip(X_train_augmented, Y_train_augmented)
    val_generator = zip(X_train_augmented_val, Y_train_augmented_val)
    print("now exiting augment_data function")
    return train_generator, val_generator,test_augmented
