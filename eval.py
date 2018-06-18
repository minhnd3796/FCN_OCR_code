from boxes import compare
import numpy as np
import json
from os import listdir

val_dir = 'validation_name_boxes'
gt_dir = '../processed/matched_json'

files = listdir(val_dir)
for file in files:
    gt_data = 