import os
import task

import skimage.io as io
import skimage.transform as transform
from skimage.morphology import label
import skimage
import numpy as np
import matplotlib.pyplot as plt
extn = task.EXTENSION
width = task.IMG_WIDTH
height = task.IMG_HEIGHT
path = f'../data/{extn}/'
image_path = f'../data/{extn}/Original/'
label_path = f'../data/{extn}/Ground Truth/'
import random


def get_images(root_path,output_size,is_label=False):
    img_paths = sorted([f'{root_path}{i}' for i in os.listdir(root_path)])
    if is_label:
        data = np.array([skimage.transform.resize(io.imread(i_path, as_gray=True),
                                                    output_shape=output_size+(1,),
                                                    mode='constant',
                                                    preserve_range=True)
                           for i_path in img_paths], dtype=np.uint8)
    else:
        data = np.array([skimage.transform.resize(io.imread(i_path)[:,:,:3],
                                                output_size,
                                                mode='constant',
                                                preserve_range=True)
                       for i_path in img_paths],dtype=np.uint8)
    data = data.astype('float32') / 255.
    return data




X_data = get_images(image_path,(height,width))
#print(X_data.shape)
Y_data = get_images(label_path,(height,width),is_label=True)
#print(Y_data.shape)

f, axarr = plt.subplots(2,4)
f.set_size_inches(20,10)
ix = random.randint(0, len(X_data[1]))
axarr[0,0].imshow(X_data[ix])
axarr[0,1].imshow(np.squeeze(Y_data[ix]))
ix = random.randint(0, len(X_data[1]))
axarr[0,2].imshow(X_data[ix])
axarr[0,3].imshow(np.squeeze(Y_data[ix]))
ix = random.randint(0, len(X_data[1]))
axarr[1,0].imshow(X_data[ix])
axarr[1,1].imshow(np.squeeze(Y_data[ix]))
ix = random.randint(0, len(X_data[1]))
axarr[1,2].imshow(X_data[ix])
axarr[1,3].imshow(np.squeeze(Y_data[ix]))

plt.show()
