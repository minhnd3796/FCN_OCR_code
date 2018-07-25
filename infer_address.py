from sys import argv
import tensorflow as tf
from FCN_OCR import inference
IMAGE_SIZE = 32
from os import listdir, mkdir, remove
from os.path import join, exists, basename
import cv2
from batch_eval_top_2 import create_patch_batch_list, batch_logits_map_inference
import numpy as np

# input_dir = '/media/minhnd/Windows10/FTI.Projects/processed/addresses/cropped_address'
# output_dir = '/media/minhnd/Windows10/FTI.Projects/processed/addresses/segmented_address'

input_dir = '/media/minhnd/Windows10/FTI.Projects/processed/addresses/lines/address_lines'
output_dir = '/media/minhnd/Windows10/FTI.Projects/processed/addresses/lines/segmented_address_lines'

CHOPPING_RATIO = 0.375
# config = tf.ConfigProto(allow_soft_placement=True)
# config.gpu_options.allow_growth = True


def infer_addr(full_filepath, filename):
    input_img = cv2.imread(full_filepath)
    if type(input_img) != type(None):
        input_batch_list, coordinate_batch_list, height, width = create_patch_batch_list(input_img=input_img,
                                                                                         batch_size=128)
        logits_map = batch_logits_map_inference(input_tensor, logits, keep_probability, sess, is_training, input_batch_list,
                                                coordinate_batch_list, height, width)
        # Inferring
        pred_annotation_map = np.array(np.argmax(logits_map, axis=2), dtype=np.uint8)
        _, thresh = cv2.threshold(pred_annotation_map, 0, 255, 0)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 2)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            for i in range(w):
                count_white = 0
                for j in range(h):
                    if thresh[y + j, x + i] == 255:
                        count_white += 1
                if count_white / h <= CHOPPING_RATIO:
                    for j in range(h):
                        pred_annotation_map[y + j, x + i] = 0

        boxes_map = np.zeros_like(pred_annotation_map, dtype=np.uint8)
        _, thresh = cv2.threshold(pred_annotation_map, 0, 255, 0)
        cv2.imwrite(join(output_dir, 'thresh_' + filename), thresh)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 2)
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > h / 1.75:
                    for i in range(w):
                        for j in range(h):
                            boxes_map[y + j, x + i] = 1
        output_words = np.transpose(np.transpose(input_img, (2, 0, 1)) * boxes_map, (1, 2, 0))
        if not exists(output_dir):
            mkdir(output_dir)
        cv2.imwrite(join(output_dir, filename), output_words)


if __name__ == '__main__':
    is_training = tf.placeholder(tf.bool, name="is_training")
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    input_tensor = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    _, logits, _ = inference(input_tensor, keep_probability, is_training)

    # sess = tf.Session(config=config)
    sess = tf.Session()
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    saver.restore(sess, argv[1])
    print(">> Restored:", argv[1])

    """ files = listdir(input_dir)
    for file in files:
        try:
            full_filepath = join(input_dir, file)
            print(full_filepath)
            infer_addr(full_filepath, file)
        except ValueError:
            remove(full_filepath) """
    infer_addr(argv[2], basename(argv[2]))
