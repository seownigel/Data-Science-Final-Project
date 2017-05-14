import os
from PIL import Image
import numpy as np

DEFAULT_PATH = "./bw_bmp_cloth/"

LABELS_DICT = {0:"dress", 1:"top", 2:"skirt", 3:"bottom", 4:"outerwear", 5:"one_piece", 6:"bodysuit"}
LABELS = range(7)
REVERSED_LABELS_DICT = dict(zip(LABELS_DICT.values(), LABELS_DICT.keys()))

IMAGE_SIZE = 100

def _read_file(filename, label_name):
    img = Image.open(filename)
    index= REVERSED_LABELS_DICT[label_name]
    x = np.asarray(img.convert("L"), dtype=float)
    y = np.asarray([0]*index +[1]+ [0]*(len(LABELS)-index-1), dtype=float)
    x = x.reshape([100, 100, 1])
    return x, y


def _read_data():
    x_y_tuple_list = []
    path = DEFAULT_PATH
    subpath_list = os.listdir(path)
    for subpath in subpath_list:
        abs_path = os.path.join(path, subpath)
        for filename in os.listdir(abs_path):
            abs_filename = os.path.join(abs_path, filename)
            x_y_tuple_list.append(_read_file(abs_filename, subpath))
    import random
    random.shuffle(x_y_tuple_list)
    return map(lambda x:x[0], x_y_tuple_list), map(lambda x: x[1], x_y_tuple_list)

xs, ys = _read_data()
training_data_size = size = len(ys) * 0.8

def get_train():
    return np.asarray(xs[:int(training_data_size)]), np.asarray(ys[:int(training_data_size)])

def deg_test():
    return np.asarray(xs[int(training_data_size):]), np.asarray(ys[int(training_data_size):])
