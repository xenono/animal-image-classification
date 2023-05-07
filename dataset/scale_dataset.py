from skimage import img_as_float
from skimage.io import imread, imsave
from skimage.transform import resize
from os import sep, listdir, mkdir, remove
from PIL import Image

img_width = 128
img_height = 128

for dataset in ["training_set, test_set"]:
    scaled_dataset = dataset + "_scaled"
    mkdir(scaled_dataset)
    for folder in listdir(dataset):
        new_folder_path = scaled_dataset + sep + folder
        folder_path = dataset + sep + folder
        mkdir(new_folder_path)
        file_counter = 0
        for file in listdir(folder_path):
            try:
                img = imread(folder_path + sep + file)

                res = resize(img, (img_width, img_height), anti_aliasing=True)
                file = file.replace(" ", "-")
                print(file)
                imsave(new_folder_path + sep + file, img_as_float(res))
                file_counter += 0
                print(str(file_counter) + ' images processed')
            except IOError:
                print("Cannot read the file, image will not be scaled")
                print("Continue...")
                continue