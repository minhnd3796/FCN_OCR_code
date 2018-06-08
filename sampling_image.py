from sys import argv
import multiprocessing as mp
import numpy as np
from cv2 import imread, imwrite
from os.path import exists, join, splitext
from os import listdir, mkdir

training_dir = "../FCN_OCR_dataset/training"
validation_dir = "../FCN_OCR_dataset/validation"
gt_dir = "../FCN_OCR_dataset/ground_truth"

input_img_dir = "../FCN_OCR_dataset/input_img"
annotation_dir = "../FCN_OCR_dataset/annotations"

CROP_SIZE = 32
STRIDE = 2
files = listdir(annotation_dir)
training_image = files[:int(len(files) * 0.8)]
validation_image = files[int(len(files) * 0.8):]
""" training_image = files[:1600]
validation_image = files[1600:2000] """

pool = mp.Pool(mp.cpu_count())

def crop_training_img(filename):
    print("Training:", filename)
    input_image = imread(join(input_img_dir, filename))
    annotation_image = imread(join(annotation_dir, filename), -1)
    width = np.shape(input_image)[1]
    height = np.shape(input_image)[0]
    i = 0
    x = 0
    while x + (CROP_SIZE - 1) <= height - 1:
        y = 0
        while y + (CROP_SIZE - 1) <= width - 1:
            i += 1
            input_image_cropped = input_image[x:x + CROP_SIZE, y:y + CROP_SIZE, :]
            imwrite(join(training_dir, splitext(filename)[0] + "_" + str(i) + ".png"), input_image_cropped)
            annotation_image_cropped = annotation_image[x:x + CROP_SIZE, y:y + CROP_SIZE]
            imwrite(join(gt_dir, splitext(filename)[0] + "_" + str(i) + ".png"), annotation_image_cropped)
            y += STRIDE
        x += STRIDE

def crop_validation_img(filename):
    print("Validation:", filename)
    input_image = imread(join(input_img_dir, filename))
    annotation_image = imread(join(annotation_dir, filename), -1)
    width = np.shape(input_image)[1]
    height = np.shape(input_image)[0]
    i = 0
    x = 0
    while x + (CROP_SIZE - 1) <= height - 1:
        y = 0
        while y + (CROP_SIZE - 1) <= width - 1:
            i += 1
            input_image_cropped = input_image[x:x + CROP_SIZE, y:y + CROP_SIZE, :]
            imwrite(join(validation_dir, splitext(filename)[0] + "_" + str(i) + ".png"), input_image_cropped)
            annotation_image_cropped = annotation_image[x:x + CROP_SIZE, y:y + CROP_SIZE]
            imwrite(join(gt_dir, splitext(filename)[0] + "_" + str(i) + ".png"), annotation_image_cropped)
            y += STRIDE
        x += STRIDE

def create_training_dataset():
    if not exists(training_dir):
        mkdir(training_dir)
    if not exists(gt_dir):
        mkdir(gt_dir)
    # pool.map(crop_training_img, training_image)
    for item in training_image:
        crop_training_img(item)
    
    return None


def create_validation_dataset():
    if not exists(validation_dir):
        mkdir(validation_dir)
    if not exists(gt_dir):
        mkdir(gt_dir)
    # pool.map(crop_validation_img, validation_image)
    for item in validation_image:
        crop_validation_img(item)
    
    return None

if __name__=="__main__":
    np.random.seed(3796)
    create_training_dataset()
    create_validation_dataset()
