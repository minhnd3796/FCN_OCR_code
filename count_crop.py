from cv2 import imread
import numpy as np
from os.path import join
from os import listdir
from sys import argv

annotation_dir = "../processed/annotations"
files = listdir(annotation_dir)

validation_image = files[int(len(files) * 0.8):]
training_image = files[:int(len(files) * 0.8)]

num_training_crops = 0
num_validation_crops = 0
CROP_SIZE = 32
STRIDE = int(argv[1])

for img_name in validation_image:
    img = imread(join('../processed/annotations', img_name))
    print("Processing", img_name)
    # num_validation_crops += (img.shape[0] - CROP_SIZE + 1) * (img.shape[1] - CROP_SIZE + 1)
    i = 0
    while i + (CROP_SIZE - 1) <= img.shape[0] - 1:
        j = 0
        while j + (CROP_SIZE - 1) <= img.shape[1] - 1:
            num_validation_crops += 1
            j += STRIDE
        i += STRIDE

for img_name in training_image:
    img = imread(join('../processed/annotations', img_name))
    print("Processing", img_name)
    # num_training_crops += (img.shape[0] - CROP_SIZE + 1) * (img.shape[1] - CROP_SIZE + 1)
    i = 0
    while i + (CROP_SIZE - 1) <= img.shape[0] - 1:
        j = 0
        while j + (CROP_SIZE - 1) <= img.shape[1] - 1:
            num_training_crops += 1
            j += STRIDE
        i += STRIDE

print("No. of training images:", num_training_crops)
print("Training size:", num_training_crops * 2497.0 / 1024.0 / 1024.0 / 1024.0)
print()
print("No. of validation images:", num_validation_crops)
print("Validation size:", num_validation_crops * 2497.0 / 1024.0 / 1024.0 / 1024.0)
print()
print("Total:", num_training_crops + num_validation_crops)
print("Total size:", (num_training_crops + num_validation_crops) * 2497.0 / 1024.0 / 1024.0 / 1024.0)