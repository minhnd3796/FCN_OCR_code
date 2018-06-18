import cv2
import tensorflow as tf
from FCN_OCR import inference
from os import environ, mkdir, listdir
from sys import argv
from batch_eval_top import create_patch_batch_list, batch_logits_map_inference
import numpy as np
from cv2 import imread, imwrite, threshold, findContours, boundingRect, rectangle, RETR_EXTERNAL, COLOR_GRAY2RGB
from os.path import exists, join, splitext
import json

IMAGE_SIZE = 32
CHOPPING_RATIO = 0.5
data_dir = '../FCN_OCR_dataset'
annotation_dir = 'annotations'
input_dir = 'input_img'

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

    num_matches = 0
    num_pixels = 0
    for file in validation_image:
        gt_annotation_map = np.array(imread(join(data_dir, annotation_dir, file), -1), dtype=np.uint8)
        input_img = imread(join(data_dir, input_dir, file))

        input_batch_list, coordinate_batch_list, height, width = create_patch_batch_list(filename=file, batch_size=128)
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

        boxes_map = np.zeros_like(pred_annotation_map, dtype=np.uint8)
        _, thresh = threshold(pred_annotation_map, 0, 255, 0)
        _, contours, _ = findContours(thresh, RETR_EXTERNAL, 2)
        json_name = splitext(file)[0] + '.json'
        json_dict = {}
        name_boxes = []
        for cnt in contours:
            x,y,w,h = boundingRect(cnt)
            name_boxes.append([x, y, x + w, y + h])
            for i in range(w):
                for j in range(h):
                    boxes_map[y + j, x + i] = 1
        json_dict['name_boxes'] = name_boxes
        print(json_dict)
        with open(join('validation_name_boxes', json_name), 'w') as outfile:
            json.dump(json_dict, outfile)

        height = pred_annotation_map.shape[0]
        width = pred_annotation_map.shape[1]
        # _, contours, _ = findContours(thresh, RETR_EXTERNAL, 2)
        num_pixels += height * width
        num_matches += np.sum(pred_annotation_map == gt_annotation_map)

        print("Locating", file + '......')
        if not exists(argv[1]):
            mkdir(argv[1])
        if not exists(join(argv[1], 'located_words')):
            mkdir(join(argv[1], 'located_words'))
        output_image = np.transpose(np.transpose(input_img, (2, 0, 1)) * boxes_map, (1, 2, 0))
        imwrite(join(argv[1], 'located_words', file), output_image)

    # Print accuracy
    print("Validation accuracy:", num_matches / num_pixels)

    """ input_batch_list, coordinate_batch_list, height, width = create_patch_batch_list(filename=argv[3], batch_size=2048)
    logits_map = batch_logits_map_inference(input_tensor, logits, keep_probability, sess, is_training, input_batch_list, coordinate_batch_list, height, width)
    pred_annotation_map = np.array(np.argmax(logits_map, axis=2), dtype=np.uint8)
    _, thresh = threshold(pred_annotation_map, 0, 255, 0)
    backtorgb = cv2.cvtColor(thresh, COLOR_GRAY2RGB)
    _, contours, _ = findContours(thresh, RETR_EXTERNAL, 2)
    for cnt in contours:
        x,y,w,h = boundingRect(cnt)
        rectangle(backtorgb, (x,y),(x+w,y+h),(0,255,0),1)
        for i in range(w):
            count_white = 0
            for j in range(h):
                if thresh[y + j, x + i] == 255:
                    count_white += 1
            if count_white / h <= CHOPPING_RATIO:
                for j in range(h):
                    thresh[y + j, x + i] = 0


    imwrite(join('/home/minhnd', argv[3]), thresh)
    imwrite(join('/home/minhnd', 'box_' + argv[3]), backtorgb)

    _, contours, _ = findContours(thresh, RETR_EXTERNAL, 2)
    backtorgb = cv2.cvtColor(thresh, COLOR_GRAY2RGB)
    print(len(contours))
    for cnt in contours:
        x,y,w,h = boundingRect(cnt)
        rectangle(backtorgb, (x,y),(x+w,y+h),(0,255,0),1)
    imwrite(join('/home/minhnd', 'box_2_' + argv[3]), backtorgb) """
