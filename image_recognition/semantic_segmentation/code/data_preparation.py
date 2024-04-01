import os


import skimage.io as io
import skimage.transform as transform

import skimage
import numpy as np
import matplotlib.pyplot as plt
extn = 'png'
width = 256
height = 256
path = f'../data/{extn}/'
image_path = f'../data/{extn}/Original/'
label_path = f'../data/{extn}/Ground Truth/'
import random

brca_rootpath = '../data/breast_cancer_histopathology/'
brca_image_path = '../data/breast_cancer_histopathology/Images/'
brca_label_path = '../data/breast_cancer_histopathology/Masks/'
def get_images(root_path,output_size,is_label=False,is_brca=False):
    img_paths = sorted([f'{root_path}{i}' for i in os.listdir(root_path)])
    print('image paths are loaded for brca')
    if is_brca:
        def is_tif_file(filename):
            filename=filename.lower()
            return filename.endswith('.tif') and '.xml' not in filename

        img_paths = sorted([f'{root_path}{i}' for i in os.listdir(root_path)])
        img_paths = list(filter(is_tif_file, img_paths))
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

X_brca_data = get_images(brca_image_path,output_size=(height,width),is_brca=True)
Y_brca_data = get_images(brca_label_path,output_size=(height,width),is_brca=True,is_label=True)
print('brca data is loaded up, now will concatenate to colorectal image set')
print('the following is the shape of the brca dataset')
print(X_brca_data.shape)
print('the following us the shape of the brca label set')
print(Y_brca_data.shape)
X_data = get_images(image_path,(height,width))

Y_data = get_images(label_path,(height,width),is_label=True)

X_data = np.concatenate((X_data,X_brca_data),axis=0)
print('shape of data after concatenating to colorectal image set')
print(X_data.shape)
Y_data = np.concatenate((Y_data,Y_brca_data),axis=0)
print('shape of labels after concatenating to colorectal image set')
print(Y_data.shape)

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
