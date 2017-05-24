import os
from PIL import Image
import numpy as np
X_AFTER_PROCESS = "./x.data"
Y_AFTER_PROCESS = "./y.data"
DEFAULT_PATH = "./bw_bmp_cloth/"
COLOUR_PATH = "./colour_images"
LABELS_DICT = {0:"dress", 1:"top", 2:"skirt", 3:"bottom", 4:"outerwear", 5:"one_piece", 6:"bodysuit"}
LABELS = range(7)
REVERSED_LABELS_DICT = dict(zip(LABELS_DICT.values(), LABELS_DICT.keys()))

IMAGE_SIZE = 100

def _read_bw_file(filename, label_name):
    img = Image.open(filename)
    index= REVERSED_LABELS_DICT[label_name]
    x = np.asarray(img.convert("L"), dtype=float)
    y = np.asarray([0]*index +[1]+ [0]*(len(LABELS)-index-1), dtype=float)
    x = x.reshape([100, 100, 1])
    return x, y

def _read_color_file(filename, label_name):
    print filename
    img = Image.open(filename)
    index= REVERSED_LABELS_DICT[label_name]
    x = np.asarray(img, dtype=float)
    y = np.asarray([0]*index +[1]+ [0]*(len(LABELS)-index-1), dtype=float)
    x = x.reshape([100, 100, 3])
    return x, y


def _read_data(force=False, path=DEFAULT_PATH):
    x_y_tuple_list = []
    subpath_list = os.listdir(path)
    for subpath in subpath_list:
        abs_path = os.path.join(path, subpath)
        for filename in os.listdir(abs_path):
            abs_filename = os.path.join(abs_path, filename)
            x_y_tuple_list.append((os.path.join(subpath, filename), _read_color_file(abs_filename, subpath)))
    import random
    random.shuffle(x_y_tuple_list)
    return map(lambda x :x [0], x_y_tuple_list), \
           map(lambda x:x[1][0], x_y_tuple_list), \
           map(lambda x: x[1][1], x_y_tuple_list)


import json

def _store_after_processed_data(filenames, xs, ys):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    with open("filename.json", "w") as fp:
        fp.write(json.dumps(filenames))
    np.save("x.data", xs)
    np.save("y.data", ys)
    pass


def _read_after_processed_data():
    xs = np.load("x.data")
    ys = np.load("y.data")
    with open("filename.json", "r") as fp:
        filenames = json.loads(fp.read())
    return filenames, xs, ys


if os.path.exists(X_AFTER_PROCESS) and os.path.exists(Y_AFTER_PROCESS):
    filenames, xs, ys = _read_after_processed_data()
    pass
else:
    filenames, xs, ys = _read_data(path=COLOUR_PATH)
    _store_after_processed_data(filenames, xs, ys)

training_data_size = size = len(ys) * 0.8

def get_train():
    return filenames[:int(training_data_size)], \
           np.asarray(xs[:int(training_data_size)]), \
           np.asarray(ys[:int(training_data_size)])

def get_test():
    return filenames[int(training_data_size):], \
           np.asarray(xs[int(training_data_size):]), \
           np.asarray(ys[int(training_data_size):])
