import random
from skimage.io import imshow
import matplotlib.pyplot as plt
def eval_plot(X_test,model):
    Y_hat = model.predict(X_test, verbose=1)
    print('model has predicted on test set')
    print(Y_hat.shape)
    idx = random.randint(0,100)
    print(X_test[idx].shape)
    imshow(X_test[idx])
    plt.show()
    imshow(Y_hat[idx][:,:,0])
    plt.show()