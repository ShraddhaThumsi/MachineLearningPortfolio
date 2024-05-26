from keras.models import load_model
import model_script
import random
from skimage.io import imshow
import matplotlib.pyplot as plt
X_test = model_script.X_test
best_model=load_model('best_model.hdf5')
Y_hat = best_model.predict(X_test, verbose=1)
print('model has predicted on test set')
print(Y_hat.shape)
idx = random.randint(0,100)
print(X_test[idx].shape)
imshow(X_test[idx])
plt.show()
imshow(Y_hat[idx][:,:,0])
plt.show()
idx = random.randint(0,100)
print(X_test[idx].shape)
imshow(X_test[idx])
plt.show()
imshow(Y_hat[idx][:,:,0])
plt.show()
idx = random.randint(0,100)
print(X_test[idx].shape)
imshow(X_test[idx])
plt.show()
imshow(Y_hat[idx][:,:,0])
plt.show()
