import cv2
import tensorflow as tf
from FCN_OCR import inference
from os import environ, mkdir, listdir
from sys import argv
from batch_eval_top import create_patch_batch_list, batch_logits_map_inference
import numpy as np
from cv2 import imread, imwrite, threshold, findContours, boundingRect, rectangle, RETR_EXTERNAL, COLOR_GRAY2RGB, namedWindow, imshow, waitKey, destroyAllWindows, contourArea
from os.path import exists, join
import json

IMAGE_SIZE = 32
CHOPPING_RATIO = 0.4375
data_dir = '../processed/addresses/new_address_lines/'
annotation_dir = 'annotations'
input_dir = 'input_img'
words_dir = 'located_words'
pred_masks_dir = 'pred_masks'
pred_boxed_masks_dir = 'pred_boxed_masks'
output_json_dir = 'validation_name_boxes'
num_matches = 0
num_pixels = 0
pad = 0

def infer_one_img(file):
    input_img = imread(join(data_dir, input_dir, file))
    input_batch_list, coordinate_batch_list, height, width = create_patch_batch_list(filename=file, batch_size=128, data_dir=data_dir)
    logits_map = batch_logits_map_inference(input_tensor, logits, keep_probability, sess, is_training, input_batch_list, coordinate_batch_list, height, width)
    # Inferring
    pred_annotation_map = np.array(np.argmax(logits_map, axis=2), dtype=np.uint8)
    _, thresh = threshold(pred_annotation_map, 0, 255, 0)
    _, contours, _ = findContours(thresh, RETR_EXTERNAL, 2)
    for cnt in contours:
        x, y, w, h = boundingRect(cnt)
        for i in range(w):
            count_white = 0
            for j in range(h):
                if thresh[y + j, x + i] == 255:
                    count_white += 1
            if count_white / h <= CHOPPING_RATIO:
                for j in range(h):
                    pred_annotation_map[y + j, x + i] = 0

    boxes_map = np.zeros_like(pred_annotation_map, dtype=np.uint8)
    _, thresh = threshold(pred_annotation_map, 0, 255, 0)
    _, contours, _ = findContours(thresh, RETR_EXTERNAL, 2)
    for cnt in contours:
        x,y,w,h = boundingRect(cnt)
        for i in range(w):
            for j in range(h):
                boxes_map[y + j, x + i] = 1
    output_words = np.transpose(np.transpose(input_img, (2, 0, 1)) * boxes_map, (1, 2, 0))
    """ namedWindow('image', cv2.WINDOW_NORMAL)
    imshow('image', output_words)
    waitKey(0)
    destroyAllWindows() """
    imwrite('/home/minhnd/Desktop/hehe.png', output_words)

def infer_img(file, model_output_dir=argv[1]):
    global num_matches
    global num_pixels
    if not exists(model_output_dir):
        mkdir(model_output_dir)

    gt_annotation_map = np.array(imread(join(data_dir, annotation_dir, file), -1), dtype=np.uint8)
    if gt_annotation_map.shape[0] >= 32 and gt_annotation_map.shape[1] >= 32:
        input_img = imread(join(data_dir, input_dir, file))

        input_batch_list, coordinate_batch_list, height, width = create_patch_batch_list(filename=file, batch_size=128, data_dir=data_dir)
        logits_map = batch_logits_map_inference(input_tensor, logits, keep_probability, sess, is_training, input_batch_list, coordinate_batch_list, height, width)

        # Inferring
        pred_annotation_map = np.array(np.argmax(logits_map, axis=2), dtype=np.uint8)
        _, thresh = threshold(pred_annotation_map, 0, 255, 0)
        _, contours, _ = findContours(thresh, RETR_EXTERNAL, 2)
        for cnt in contours:
            x,y,w,h = boundingRect(cnt)
            for i in range(w):
                count_white = 0
                for j in range(h):
                    if thresh[y + j, x + i] == 255:
                        count_white += 1
                if count_white / h <= CHOPPING_RATIO:
                    for j in range(h):
                        pred_annotation_map[y + j, x + i] = 0
        if not exists(join(model_output_dir, pred_masks_dir)):
            mkdir(join(model_output_dir, pred_masks_dir))
        _, thresh = threshold(pred_annotation_map, 0, 255, 0)
        imwrite(join(model_output_dir, pred_masks_dir, file), thresh)

        boxes_map = np.zeros_like(pred_annotation_map, dtype=np.uint8)
        _, thresh = threshold(pred_annotation_map, 0, 255, 0)
        _, contours, _ = findContours(thresh, RETR_EXTERNAL, 2)
        json_name = file.replace('png', 'json')
        json_dict = {}
        name_boxes = []
        for cnt in contours:
            if contourArea(cnt) > 200:
                x,y,w,h = boundingRect(cnt)
                input_height, input_width, _ = input_img.shape
                # x = max(0, x - pad)
                y = max(0, y - pad)
                # end_x = min(x + w + pad, input_width)
                end_x = x + w
                end_y = min(y + h + pad, input_height)
                # w = end_x - x
                h = end_y - y
                if w > h / 1.75:
                    name_boxes.append([x, y, end_x, end_y])
                    for i in range(w):
                        for j in range(h):
                            boxes_map[y + j, x + i] = 1
        _, thresh_boxes = threshold(boxes_map, 0, 255, 0)
        if not exists(join(model_output_dir, pred_boxed_masks_dir)):
            mkdir(join(model_output_dir, pred_boxed_masks_dir))
        imwrite(join(model_output_dir, pred_boxed_masks_dir, file), thresh_boxes)
        json_dict['name_boxes'] = name_boxes
        print(json_dict)
        if not exists(output_json_dir):
            mkdir(output_json_dir)
        with open(join(output_json_dir, json_name), 'w') as outfile:
            json.dump(json_dict, outfile)

        height = pred_annotation_map.shape[0]
        width = pred_annotation_map.shape[1]
        # _, contours, _ = findContours(thresh, RETR_EXTERNAL, 2)
        num_pixels += height * width
        num_matches += np.sum(pred_annotation_map == gt_annotation_map)

        output_words = np.transpose(np.transpose(input_img, (2, 0, 1)) * boxes_map, (1, 2, 0))
        if not exists(join(model_output_dir, words_dir)):
            mkdir(join(model_output_dir, words_dir))
        imwrite(join(model_output_dir, words_dir, file), output_words)

if __name__ == '__main__':
    environ["CUDA_VISIBLE_DEVICES"] = argv[2]

    is_training = tf.placeholder(tf.bool, name="is_training")
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    input_tensor = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    _, logits, _ = inference(input_tensor, keep_probability, is_training)

    sess = tf.Session()
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    saver.restore(sess, argv[1])
    print(">> Restored:", argv[1])

    files = listdir(join(data_dir, annotation_dir))
    validation_image = files[int(len(files) * 0.8):]
    for file in validation_image:
        print("Locating", file)
        infer_img(file)
    print("Validation accuracy:", num_matches / num_pixels)

    # infer_one_img(argv[3])
