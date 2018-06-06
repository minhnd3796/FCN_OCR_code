import tensorflow as tf
from FCN_OCR import inference
from os import environ, mkdir
from sys import argv
from batch_eval_top import create_patch_batch_list, batch_logits_map_inference
import numpy as np
from cv2 import imread, imwrite
from os.path import exists, join

IMAGE_SIZE = 32

if __name__ == '__main__':
    environ["CUDA_VISIBLE_DEVICES"] = argv[2]    

    is_training = tf.placeholder(tf.bool, name="is_training")
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    input_tensor = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 2], name="input_image")
    _, logits, _ = inference(input_tensor, keep_probability, is_training)

    sess = tf.Session()
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Init logits maps

    # Accumulate logits maps
    ckpt = tf.train.get_checkpoint_state(argv[1])
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    print(">> Restored:", ckpt.model_checkpoint_path)
    for i in range(num_img_files):
        print(ckpt_path, "inferring", filename[i])
        input_batch_list, coordinate_batch_list, height, width = create_patch_batch_list(filename=argv[3], batch_size=512, num_channels=6)
        current_logits_map = batch_logits_map_inference(input_tensor, logits, keep_probability, sess, is_training, input_batch_list, coordinate_batch_list, height, width)
        logits_maps[i] += current_logits_map

    # Inferring
    for i in range(num_img_files):
        pred_annotation_maps[i] = np.argmax(logits_maps[i], axis=2)
        # num_matches += np.sum(pred_annotation_maps[i] == gt_annotation_maps[i]) # Comment if for submission
        height = pred_annotation_maps[i].shape[0]
        width = pred_annotation_maps[i].shape[1]
        output_image = np.zeros((height, width, 3), dtype=np.uint8)

        print("Generating", filename[i] + '.png......')
        for y in range(height):
            for x in range(width):
                if pred_annotation_maps[i][y, x]==1:
                    output_image[y, x] = 255
        if not exists(join(argv[1], 'predicted_mask')):
            mkdir(join(argv[1], 'predicted_mask'))
        """ if not exists(join(argv[1], 'submission_PIL')):
            mkdir(join(argv[1], 'submission_PIL'))
        img = Image.fromarray(output_image)
        b, g, r = img.split()
        img = Image.merge("RGB", (r, g, b))
        img.save(join(argv[1], 'submission_PIL', filename[i] + '_class.tif')) """
        imwrite(join(argv[1], 'predicted_mask', filename[i] + '_class.png'), output_image)

    # Print ensemble accuracy
    # print("Ensembled Validation Accuracy:", num_matches / num_pixels) # Comment if for submission
