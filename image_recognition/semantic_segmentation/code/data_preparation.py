import os
import skimage.io as io
import skimage.transform as transform
import skimage
import numpy as np
import matplotlib.pyplot as plt
from medpy.io import load
import random
extn = 'png'
width = 256
height = 256
path = f'../data/{extn}/'
image_path = f'../data/{extn}/Original/'
label_path = f'../data/{extn}/Ground Truth/'



brca_rootpath = '../data/breast_cancer_histopathology/'
brca_image_path = f'{brca_rootpath}Images/'
brca_label_path = f'{brca_rootpath}Masks/'

hippo_rootpath = '../data/hippocampus/'
hippo_image_path = f'{hippo_rootpath}imagesTr_original_data/'
hippo_label_path = f'{hippo_rootpath}labelsTr_original_data/'

heart_rootpath = '../data/heart/'
heart_image_path = f'{heart_rootpath}imagesTr/'
heart_label_path = f'{heart_rootpath}labelsTr/'

prostrate_rootpath = '../data/prostrate/'
prostrate_image_path = f'{prostrate_rootpath}imagesTr/'
prostrate_label_path = f'{prostrate_rootpath}labelsTr/'

spleen_rootpath = '../data/spleen/'
spleen_image_path = f'{spleen_rootpath}imagesTr/'
spleen_label_path = f'{spleen_rootpath}labelsTr/'


def process_brca(root_path):
    def is_tif_file(filename):
        filename = filename.lower()
        return filename.endswith('.tif') and '.xml' not in filename

    img_paths = sorted([f'{root_path}{i}' for i in os.listdir(root_path)])

    img_paths = list(filter(is_tif_file, img_paths))

    return img_paths
def unzip_and_process_nii_files(root_path):
    img_paths = sorted([f'{root_path}{i}' for i in os.listdir(root_path)])
    def unzip_file(filename):
        os.system('gunzip ' + filename)
    map(unzip_file,img_paths)

    def take_only_nii(filename):
        return filename.endswith('.nii') and '.gz' not in filename
    img_paths = sorted(list(filter(take_only_nii, img_paths)))

    def convert_file_to_png(filename):
        image_data, _ = load(filename)
        normalized_data = image_data / np.max(image_data)
        cmap = plt.get_cmap('viridis')
        rgb_data = cmap(normalized_data)[:, :, :3]
        rgb_data = (rgb_data * 255.).astype(np.uint8)
        png_x = filename.replace('.nii', '.png')

        png_filename = f'{png_x}'
        plt.imsave(png_filename, rgb_data)

        return png_x

    img_paths = list(map(convert_file_to_png, img_paths))
    return img_paths
def process_nii_files(root_path):

    img_paths = unzip_and_process_nii_files(root_path)

    return img_paths

def get_images(root_path,output_size,is_label=False,tissue_type='colorectal'):
    img_paths = sorted([f'{root_path}{i}' for i in os.listdir(root_path)])
    """allowed options for tissue type: 
        'colorectal'
        'breast cancer'
        'hippocampus'
        'heart'
        'prostrate'
        'spleen'
    """
    if tissue_type=='breast cancer':

        #parses the breast cancer dataset. The folder contains only tif and .xml files so we have to read it along with cleaning up file names
        img_paths = process_brca(root_path)

    if tissue_type in ['hippocampus','heart','prostrate','spleen']:

        img_paths = process_nii_files(root_path)



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
X_brca_data = get_images(brca_image_path,output_size=(height,width),tissue_type='breast cancer')
Y_brca_data = get_images(brca_label_path,output_size=(height,width),tissue_type='breast cancer',is_label=True)
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
X_hippo_data = get_images(hippo_image_path,(height,width),tissue_type='hippocampus')
Y_hippo_data = get_images(hippo_label_path,(height,width),tissue_type='hippocampus',is_label=True)
print('shape of hippocampus data')
print(X_hippo_data.shape)
print(Y_hippo_data.shape)

#importing heart images
X_heart_data = get_images(heart_image_path,(height,width),tissue_type='heart')
Y_heart_data = get_images(heart_label_path,(height,width),tissue_type='heart',is_label=True)
print('shape of heart data')
print(X_heart_data.shape)
print(Y_heart_data.shape)

#importing prostrate images
X_prostrate_data = get_images(prostrate_image_path,(height,width),tissue_type='prostrate')
Y_prostrate_data = get_images(prostrate_label_path,(height,width),tissue_type='prostrate',is_label=True)
print('shape of prostrate data')
print(X_prostrate_data.shape)
print(Y_prostrate_data.shape)

#importing spleen images
X_spleen_data = get_images(spleen_image_path,(height,width),tissue_type='spleen')
Y_spleen_data = get_images(spleen_label_path,(height,width),tissue_type='spleen',is_label=True)

#appending for larger dataset
X_data = np.concatenate((X_data,X_brca_data,X_hippo_data,X_heart_data,X_prostrate_data,X_spleen_data),axis=0)
Y_data = np.concatenate((Y_data,Y_brca_data,Y_hippo_data,Y_heart_data,Y_prostrate_data,Y_spleen_data),axis=0)
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
