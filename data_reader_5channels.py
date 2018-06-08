import os
import glob
from os.path import exists, join
import random

from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile

def read_dataset_OCR(image_dir):
    pickle_filename = "OCR.pickle"
    pickle_filepath = join(image_dir, pickle_filename)
    if not exists(pickle_filepath):
        result = create_image_list_OCR(image_dir)
        print("pickling...")
        with open(pickle_filepath, "wb") as f:
            pickle.dump(result,f,pickle.HIGHEST_PROTOCOL)
    else:
        print("pickle file found")
    with open(pickle_filepath,"rb") as f:
        result = pickle.load(f)
        training_records = result['tiny_training']
        validation_records = result['tiny_validation']
        del result
    return training_records, validation_records


def create_image_list_OCR(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    directories = ['tiny_training', 'tiny_validation']
    image_list = {}
    for directory in directories:
        file_list = []
        image_list[directory] = []
        file_glob = join(image_dir, directory, "*." + "png")
        file_list.extend(glob.glob(file_glob))
        if not file_list:
            print("no files found")
        else:
            for f in file_list:
                filename = os.path.splitext(f.split("/")[-1])[0]
                annotation_file = join(image_dir, "tiny_ground_truth", filename + ".png")
                if exists(annotation_file):
                    record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                    image_list[directory].append(record)
                else:
                    print("Annotation file not found for %s - Skipping" % filename)
        random.shuffle(image_list[directory])
        num_of_images = len(image_list[directory])
        print('No. of %s files: %d' % (directory, num_of_images))
    return image_list
