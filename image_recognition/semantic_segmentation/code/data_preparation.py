import os
import task

import skimage.io as io
import skimage.transform as transform
from skimage.morphology import label
import skimage
import numpy as np
extn = task.EXTENSION
width = task.IMG_WIDTH
height = task.IMG_HEIGHT
path = f'../data/{extn}/'
image_path = f'../data/{extn}/Original/'
label_path = f'../data/{extn}/Ground Truth/'


def get_images(root_path,output_size,is_label=False):
    img_paths = sorted([f'{root_path}{i}' for i in os.listdir(root_path)])
    if is_label:
        data = np.array([skimage.transform.resize(io.imread(i_path, as_gray=True),
                                                    output_size,
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
print(X_data.shape)
Y_data = get_images(label_path,(height,width),is_label=True)
print(Y_data.shape)



