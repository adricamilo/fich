import os
from fich import _get_name as get_name
from fich import get_image_paths, load_cv2_list, rotate_cv2_list
import cv2
import random


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


random.seed(130568209244504297)
image_folder = os.path.join("fich_database", "scanned")
input_folder = os.path.join("fich_database", "inputs_scanned")

image_list = get_image_paths(image_folder)
random.shuffle(image_list)

n = len(image_list)
input_lists = {
    90: [],
    270: [],
    0: []
}

for i in range(n):
    if 3 * i < n:
        input_lists[90].append(i)
        input_lists[0].append(i)
    elif 3 * i < 2 * n:
        input_lists[270].append(i)
        input_lists[90].append(i)
    else:
        input_lists[0].append(i)
        input_lists[270].append(i)

to_words = {
    90: "clockwise",
    270: "counterclockwise",
    0: "upright"
}

cv2_list = load_cv2_list(image_list)

for orient, id_list in input_lists.items():
    print(f"Working on {to_words[orient]} folder...")
    folder = os.path.join(input_folder, to_words[orient])
    create_dir(folder)
    im_counter = 0
    sub_list = [cv2_list[i] for i in id_list]
    random.shuffle(sub_list)
    for im in rotate_cv2_list(sub_list, orient):
        path = os.path.join(folder, get_name(im_counter, "image", ".jpg"))
        cv2.imwrite(path, im)
        im_counter = im_counter + 1
    print(f"Finished {to_words[orient]} folder")
