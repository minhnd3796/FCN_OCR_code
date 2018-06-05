from os import listdir

import numpy as np
from cv2 import imread

BASE_DIRECTORY = '../FCN_OCR_dataset/'
BASE_DIRECTORY = '../processed/'

b_mean = []
g_mean = []
r_mean = []
for filename in listdir(BASE_DIRECTORY+'/matched_fullname'):
    print("Reading:", filename)
    image = imread(BASE_DIRECTORY+'/matched_fullname/'+filename)
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]
    b_mean.append(np.mean(b))
    g_mean.append(np.mean(g))
    r_mean.append(np.mean(r))
print('B:', np.mean(np.array(b_mean)))
print('G:', np.mean(np.array(g_mean)))
print('R:', np.mean(np.array(r_mean)))
