from PIL import Image
import glob
import tqdm
import numpy as np
import cv2


def get_pixel_data(file_path, num_samples=650, silence=False):
    file_names = glob.glob(file_path + "*.jpg")
    if num_samples > 0:
        file_names = file_names[:num_samples]
    photo_values = []
    im = Image.open(file_names[0], 'r')
    width, height = im.size
    for file_name in tqdm.tqdm(file_names, disable=silence):
        im = Image.open(file_name, 'r')
        pix = np.asarray(im)
        photo_values.append(pix)
    return(width, height, photo_values)


def get_pixel_data_hog(file_path, num_samples=650):
    file_names = glob.glob(file_path + "*.jpg")
    if num_samples > 0:
        file_names = file_names[:num_samples]
    photo_values = []
    hog_values = []
    im = Image.open(file_names[0], 'r')
    width, height = im.size
    for file_name in tqdm.tqdm(file_names):
        im = Image.open(file_name, 'r')
        pix = np.asarray(im)
        photo_values.append(pix)
        hog = cv2.HOGDescriptor()
        image = cv2.imread(file_name)
        hog_im = hog.compute(image)
        hog_values.append(hog_im)
    return(width, height, photo_values, hog_values)


def create_legend(img, pt1, pt2):
    text1 = "Before resampling"
    cv2.putText(img, text1, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    text2 = "After resampling"
    cv2.putText(img, text2, pt2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
