import os
image_path = '../prostrate/imagesTr/'
label_path = '../prostrate/labelsTr/'


image_names = os.listdir(image_path)
label_names = os.listdir(label_path)
for image_name in label_names:
    print(image_name)
    os.system('gunzip ' + os.path.join(label_path,image_name))
# for label_name in label_names:
#     os.system('gunzip ' + label_name)
