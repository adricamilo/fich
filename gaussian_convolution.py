from fich import _gaussian_filter as gaussian_filter
import tkinter as tk
import cv2
from PIL import ImageTk, Image
from numpy import ndarray

INITIAL_BLOCK_SIZE = 101
MINIMUM_BLOCK_SIZE = 3
MAXIMUM_BLOCK_SIZE = 201
INITIAL_C = 31
MINIMUM_C = 0
MAXIMUM_C = 60

image_path = ""


def filter_image(image: ndarray, block_size: int, c: int):
    display_image = Image.fromarray(
            cv2.cvtColor(
                gaussian_filter(image, block_size, c),
                cv2.COLOR_BGR2RGB
            )
        )
    max_dim = 1000
    width, height = display_image.size
    ratio = min(max_dim/width, max_dim/height)
    return display_image.resize((int(width * ratio), int(height * ratio)))


def update_display(value):
    new_block_size = block_size_scale.get()
    new_c = c_scale.get()
    new_image = filter_image(original_image, new_block_size, new_c)
    new_image_tk = ImageTk.PhotoImage(new_image)
    label.configure(image=new_image_tk)
    label.image = new_image_tk


window = tk.Tk()
window.title("Gaussian convolution")

frame = tk.Frame(window)
frame.pack(anchor=tk.CENTER)

original_image = cv2.imread(image_path)
filtered = filter_image(original_image, INITIAL_BLOCK_SIZE, INITIAL_C)
filtered_tk = ImageTk.PhotoImage(filtered)

label = tk.Label(frame, image=filtered_tk)
label.grid(row=0, column=0)

scales = tk.Frame(frame)
scales.grid(row=0, column=1)

shape = tk.Label(scales, text=f"Image size = {original_image.shape[0]}×{original_image.shape[1]}")
shape.grid(row=0, column=0)

block_size = tk.IntVar()
c = tk.IntVar()

block_size_scale = tk.Scale(scales,
                            variable=block_size,
                            from_=MINIMUM_BLOCK_SIZE,
                            to=MAXIMUM_BLOCK_SIZE,
                            resolution=2,
                            orient=tk.HORIZONTAL,
                            length=400,
                            label="block size",
                            command=update_display)
block_size_scale.set(INITIAL_BLOCK_SIZE)
block_size_scale.grid(row=1, column=0)

c_scale = tk.Scale(scales,
                   variable=c,
                   from_=MINIMUM_C,
                   to=MAXIMUM_C,
                   resolution=1,
                   orient=tk.HORIZONTAL,
                   length=400,
                   label="c",
                   command=update_display)
c_scale.set(INITIAL_C)
c_scale.grid(row=2, column=0)

window.mainloop()
