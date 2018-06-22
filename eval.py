from boxes import compare
import numpy as np
import json
from os import listdir, mkdir
from os.path import join, exists
from shutil import copy

pred_dir = 'validation_name_boxes'
gt_dir = '../processed/matched_json'
cropped_dir = '../logs-FCN-OCR/model.ckpt-2/located_words/'
full_img_with_boxes_dir = '../processed/matched_img/'
wrong_img_correct_dir = 'wrong_images_correct'
wrongly_cropped_correct_dir = 'wrongly_cropped_correct'
wrong_img_recall_dir = 'wrong_images_recall'
wrongly_cropped_recall_dir = 'wrongly_cropped_recall'

files = listdir(pred_dir)
count_correct = 0
count_recall = 0

for file in files:
    with open(join(gt_dir, file)) as gt_file:
        gt_data = json.load(gt_file)
    json_gt_name_boxes = gt_data['name_boxes']
    fullname_box = gt_data['full_name_boxes']
    gt_name_boxes = []
    for json_gt_name_box in json_gt_name_boxes:
        gt_name_box = [0, 0, 0, 0]
        gt_name_box[0] = json_gt_name_box[0] - fullname_box[0]
        
        if gt_name_box[0] < 0:
            gt_name_box[0] = 0

        gt_name_box[1] = json_gt_name_box[1] - fullname_box[1]
        if gt_name_box[1] < 0:
            gt_name_box[1] = 0

        gt_name_box[2] = json_gt_name_box[2] - fullname_box[0]
        if gt_name_box[2] > fullname_box[2] - fullname_box[0]:
            gt_name_box[2] = fullname_box[2] - fullname_box[0]

        gt_name_box[3] = json_gt_name_box[5] - fullname_box[1]
        if gt_name_box[3] > fullname_box[3] - fullname_box[1]:
            gt_name_box[3] = fullname_box[3] - fullname_box[1]
        gt_name_boxes.append(gt_name_box)

    with open(join(pred_dir, file)) as pred_file:
        pred_data = json.load(pred_file)
        pred_name_boxes = pred_data['name_boxes']
        correct, recall, hits = compare(pred_name_boxes, gt_name_boxes)
        
        if correct == True:
            count_correct += 1
        else:
            if not exists(wrong_img_correct_dir):
                mkdir(wrong_img_correct_dir)
            if not exists(wrongly_cropped_correct_dir):
                mkdir(wrongly_cropped_correct_dir)
            img_name = file.replace('json', 'png')
            copy(join(full_img_with_boxes_dir, img_name), join(wrong_img_correct_dir, img_name))
            copy(join(cropped_dir, img_name), join(wrongly_cropped_correct_dir, img_name))
        if recall == True:
            count_recall += 1
        else:
            if not exists(wrong_img_recall_dir):
                mkdir(wrong_img_recall_dir)
            if not exists(wrongly_cropped_recall_dir):
                mkdir(wrongly_cropped_recall_dir)
            img_name = file.replace('json', 'png')
            copy(join(full_img_with_boxes_dir, img_name), join(wrong_img_recall_dir, img_name))
            copy(join(cropped_dir, img_name), join(wrongly_cropped_recall_dir, img_name))
    
print("Correct:", count_correct / len(files))
print("Recall:", count_recall / len(files))