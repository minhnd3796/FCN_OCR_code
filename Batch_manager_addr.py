import numpy as np
from cv2 import imread, imwrite
import tensorflow as tf
from os.path import exists, join
from os import mkdir
from batch_eval_top import eval_dir
# from batch_eval_potsdam import eval_dir_potsdam
# from batch_eval_15 import eval_dir_15

STRIDE = 2
CROP_SIZE = 32

class Batch_manager:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0
    seed = 3796
    data_dir = ''
    input_img_dir = 'input_img'
    annotation_dir = 'annotations'

    def __init__(self, records_list, data_dir, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list
        self.image_options = image_options
        self.data_dir = data_dir
        self._read_images()

    def get_crops(self, img_type_dir):
        imgs = []
        for file in self.files:
            if img_type_dir == self.annotation_dir:
                full_img = imread(join(self.data_dir, img_type_dir, file), -1)
            else:
                full_img = imread(join(self.data_dir, img_type_dir, file))
            width = np.shape(full_img)[1]
            height = np.shape(full_img)[0]
            i = 0
            x = 0
            while x + (CROP_SIZE - 1) <= height - 1:
                y = 0
                while y + (CROP_SIZE - 1) <= width - 1:
                    i += 1
                    if img_type_dir == self.annotation_dir:
                        imgs.append(full_img[x:x + CROP_SIZE, y:y + CROP_SIZE])
                    else:
                        imgs.append(full_img[x:x + CROP_SIZE, y:y + CROP_SIZE, :])
                    y += STRIDE
                x += STRIDE
        imgs = np.array(imgs)
        if img_type_dir == self.annotation_dir:
            imgs = np.expand_dims(imgs, axis=3)
        return imgs

    def _read_images(self):
        self.__channels = True
        # self.images = np.array([imread(filename['image']).astype(np.float16) for filename in self.files])
        self.images = np.array(self.get_crops(self.input_img_dir))
        self.__channels = False
        # self.annotations = np.array([np.expand_dims(self._transform_annotations(filename['annotation']), axis=3) for filename in self.files])
        self.annotations = np.array(self.get_crops(self.annotation_dir))
        print(self.images.shape)
        print(self.annotations.shape)

    def _transform(self, filename):
        image = np.load(filename)
        if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])
        return np.array(image).astype(np.float16)

    def _transform_annotations(self, filename):
        return imread(filename, -1)

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, saver, batch_size, input_tensor, logits, keep_probability, sess, is_training, log_dir, encoding_keep_prob=None, is_validation=False):
        start = self.batch_offset
        self.batch_offset += batch_size
        np.random.seed(self.seed)
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            saver.save(sess, log_dir + "model.ckpt", self.epochs_completed)
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            """ if not is_validation:
                eval_dir(input_tensor, logits, keep_probability, sess, is_training, batch_size, log_dir, self.epochs_completed, encoding_keep_prob=encoding_keep_prob, is_validation=False, num_channels=3)
                eval_dir(input_tensor, logits, keep_probability, sess, is_training, batch_size, log_dir, self.epochs_completed, encoding_keep_prob=encoding_keep_prob, is_validation=True, num_channels=3) """
            # Start next epoch
            start = 0
            self.batch_offset = batch_size
        if start == 0:
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
        end = self.batch_offset
        return self.images[start:end].astype(dtype=np.float32), self.annotations[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes].astype(dtype=np.float32), self.annotations[indexes]
