import os

import nibabel
import skimage.io as io
import skimage.transform as transform
import imageio
import skimage
import numpy as np
import matplotlib.pyplot as plt
from medpy.io import load
extn = 'png'
width = 256
height = 256
path = f'../data/{extn}/'
image_path = f'../data/{extn}/Original/'
label_path = f'../data/{extn}/Ground Truth/'
import random
from PIL import Image

brca_rootpath = '../data/breast_cancer_histopathology/'
brca_image_path = '../data/breast_cancer_histopathology/Images/'
brca_label_path = '../data/breast_cancer_histopathology/Masks/'

hippo_rootpath = '../data/hippocampus/'
hippo_image_path = '../data/hippocampus/imagesTr_original_data/'
hippo_label_path = '../data/hippocampus/labelsTr_original_data/'

def process_brca(root_path):
    def is_tif_file(filename):
        filename = filename.lower()
        return filename.endswith('.tif') and '.xml' not in filename

    img_paths = sorted([f'{root_path}{i}' for i in os.listdir(root_path)])
    print('image paths are loaded for brca')
    img_paths = list(filter(is_tif_file, img_paths))

    return img_paths
def process_hippo(root_path):
    hippo_image_paths = sorted([f'{root_path}{i}' for i in os.listdir(root_path)])
    def is_nii(filename):
        filename = filename.lower()
        return filename.endswith('.nii')

    img_paths = list(filter(is_nii, hippo_image_paths))


    def convert_file_to_png(filename):
        image_data,_ = load(filename)
        print("inside convert to png func")
        print("checking if image is 3d or 2d")
        print(len(image_data.shape))
        print("dimensions of the image are: ")
        print(f'{image_data.shape[0]} x {image_data.shape[1]}')

        return image_data

    img_paths = list(map(convert_file_to_png, img_paths))
    print('i have converted the images to png, now i want to see how many images are there')
    print(type(img_paths))
    print(len(img_paths))



    return img_paths

def get_images(root_path,output_size,is_label=False,is_brca=False, is_hippocampus=False):
    img_paths = sorted([f'{root_path}{i}' for i in os.listdir(root_path)])

    if is_brca:

        #parses the breast cancer dataset. The folder contains only tif and .xml files so we have to read it along with cleaning up file names
        img_paths = process_brca(root_path)

    if is_hippocampus:

        img_paths = process_hippo(root_path)





    if is_label:
        #if it's the label dataset then we have to take it as a grayscale image, therefore only one channel is necessary
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

#importing breast cancer images
X_brca_data = get_images(brca_image_path,output_size=(height,width),is_brca=True)
Y_brca_data = get_images(brca_label_path,output_size=(height,width),is_brca=True,is_label=True)
print('shape of brca data')
print(X_brca_data.shape)
print(Y_brca_data.shape)
#importing colorectal images
X_data = get_images(image_path,(height,width))
Y_data = get_images(label_path,(height,width),is_label=True)
print('shape of colorectal data')
print(X_data.shape)
print(Y_data.shape)
#importing hippocampal images
X_hippo_data = get_images(hippo_image_path,(height,width),is_hippocampus=True)
Y_hippo_data = get_images(hippo_label_path,(height,width),is_hippocampus=True,is_label=True)
print('shape of hippocampus data')
print(X_hippo_data.shape)
print(Y_hippo_data.shape)

#appending for larger dataset
X_data = np.concatenate((X_data,X_brca_data,X_hippo_data),axis=0)
Y_data = np.concatenate((Y_data,Y_brca_data,Y_hippo_data),axis=0)
print('shape of full data after concatenating various cancers data')
print(X_data.shape)
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
