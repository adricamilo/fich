import os
import fich
import cv2
import random

random.seed(130568209244504297)
image_folder = os.path.join("fich_database", "images")
input_folder = os.path.join("fich_database", "inputs")

image_list = fich.get_image_paths(image_folder)
random.shuffle(image_list)

n = len(image_list)
input_lists = {
    90: [],
    270: [],
    0: []
}

for i, image in enumerate(image_list):
    if 3 * i < n:
        input_lists[90].append(image)
    elif 3 * i < 2 * n:
        input_lists[270].append(image)
    else:
        input_lists[0].append(image)

to_words = {
    90: "clockwise",
    270: "counterclockwise",
    0: "upright"
}

for orient, im_list in input_lists.items():
    print(f"Working on {to_words[orient]} folder...")
    cv2_list = fich.load_cv2_list(im_list, orient)
    print("Rotated...")
    im_counter = 0
    for im in cv2_list:
        folder = os.path.join(input_folder, to_words[orient])
        cv2.imwrite(os.path.join(folder, fich._get_name(im_counter, "image", ".jpg")), im)
        im_counter = im_counter + 1
    print(f"Finished {to_words[orient]} folder")
