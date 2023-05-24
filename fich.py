from collections.abc import Iterable
import os
from PIL import Image
import cv2
import pytesseract
import tempfile
import pdf2image
import numpy
from numpy import ndarray
from matplotlib import pyplot as plt
from unidecode import unidecode
import json
from pypdf import PdfReader, PdfWriter
import fitz
import cnn_eval


def _is_image(path: str) -> bool:
    has_image_format = path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
    not_garbage = not os.path.basename(path).startswith("._")
    return has_image_format and not_garbage


def get_image_paths(folder, alphabetic: bool = True) -> list[str]:
    images = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if os.path.isfile(path) and _is_image(path):
            images.append(file)

    if alphabetic:
        images.sort()

    return [os.path.join(folder, image) for image in images]


def rotate_pil_list(images: list[Image.Image], degrees: int) -> list[Image.Image]:
    degrees = degrees % 360

    if degrees == 0:
        return images

    rotations = {
        90: Image.ROTATE_90,
        180: Image.ROTATE_180,
        270: Image.ROTATE_270
    }

    try:
        images = [image.transpose(rotations[degrees]) for image in images]
    except KeyError:
        raise ValueError("degrees must be 0, 90, 180 or 270.")

    return images


def load_pil_list(images: list, degrees: int = 0) -> list[Image.Image]:
    loaded = [Image.open(image) for image in images]

    return rotate_pil_list(loaded, degrees)


def _get_name(counter: int, name: str, extension: str) -> str:
    return name + "[" + str(counter) + "]" + extension


def _save_name(path: str, base: str = "new_file", extension: str = ".pdf") -> str:
    if not os.path.isdir(path):
        return path

    counter = 0
    base_name = _get_name(counter, base, extension)
    while True:
        if os.path.exists(os.path.join(path, base_name)):
            counter += 1
            base_name = _get_name(counter, base, extension)
            continue

        return os.path.join(path, base_name)


def pil_list_to_pdf(images: list[Image.Image], path: str) -> str:
    filename = _save_name(path, "joined")
    images[0].save(filename, "PDF",
                   save_all=True, append_images=images[1:])
    return filename


def rotate_cv2_list(images: list[ndarray], degrees: int) -> list[ndarray]:
    degrees = degrees % 360

    if degrees == 0:
        return images

    rotations = {
        90: cv2.ROTATE_90_COUNTERCLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_CLOCKWISE
    }

    try:
        images = [cv2.rotate(image, rotations[degrees]) for image in images]
    except KeyError:
        raise ValueError("degrees must be 0, 90, 180 or 270.")

    return images


def load_cv2_list(images: list, degrees: int = 0) -> list[ndarray]:
    # Loads image in BGR format by default
    loaded = [cv2.imread(image) for image in images]

    return rotate_cv2_list(loaded, degrees)


def _gaussian_filter(image: ndarray, block_size: int, c: int) -> ndarray:
    if block_size % 2 == 0:
        raise ValueError("block_size must be odd.")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    filtered = cv2.adaptiveThreshold(
        gray_image,
        255,  # maximum value assigned to pixel values exceeding the threshold
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # gaussian weighted sum of neighborhood
        cv2.THRESH_BINARY,  # thresholding type
        block_size,  # block size
        c  # subtracted from the mean or weighted sum of the neighbourhood
    )

    return filtered


def gaussian_cv2_list(images: list[ndarray], block_size: int, c: int) -> list[ndarray]:
    filtered = [_gaussian_filter(image, block_size, c) for image in images]

    return filtered


def cv2_to_pil_list(images: list[ndarray]) -> list[Image.Image]:
    images_pillow = [Image.fromarray(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for image in images]

    return images_pillow


def pil_to_cv2_list(images: list[Image.Image]) -> list[ndarray]:
    # noinspection PyTypeChecker
    images_cv2 = [cv2.cvtColor(numpy.array(image.convert('RGB')),
                               cv2.COLOR_RGB2BGR) for image in images]

    return images_cv2


def pil_list_to_ocr(images: list[Image.Image], path: str, language: str = "spa") -> str:
    temp_dir = tempfile.TemporaryDirectory()
    files = []
    for i, image in enumerate(images):
        image_name = os.path.join(temp_dir.name, str(i) + ".tiff")
        image.save(image_name)
        files.append(image_name + "\n")

    images_file = os.path.join(temp_dir.name, "images.txt")
    with open(images_file, "w") as f:
        f.writelines(files)

    pdf = pytesseract.image_to_pdf_or_hocr(images_file, extension="pdf",
                                           lang=language)

    filename = _save_name(path, "generated")

    with open(filename, "w+b") as f:
        f.write(pdf)  # pdf type is bytes by default

    temp_dir.cleanup()

    return filename


def pdf_to_ocr(path: str, language: str = "spa") -> str:
    if not (os.path.isfile(path) and path.endswith(".pdf")):
        raise ValueError("path argument must be the path to a PDF file.")

    with tempfile.TemporaryDirectory() as folder:
        images = pdf2image.convert_from_path(path, output_folder=folder)
        ocr_file = os.path.splitext(path)[0] + "[OCR].pdf"
        filename = pil_list_to_ocr(images, ocr_file, language)

    return filename


def pdf_to_filtered(path: str, block_size: int, c: int) -> str:
    if not (os.path.isfile(path) and path.endswith(".pdf")):
        raise ValueError("path argument must be the path to a PDF file.")

    with tempfile.TemporaryDirectory() as folder:
        pil_images = pdf2image.convert_from_path(path, output_folder=folder)
        cv2_filtered = gaussian_cv2_list(pil_to_cv2_list(pil_images), block_size, c)
        pil_filtered = cv2_to_pil_list(cv2_filtered)
        filtered_name = os.path.splitext(path)[0] + f"[f{block_size},{c}].pdf"

        filename = pil_list_to_pdf(pil_filtered, filtered_name)

    return filename


def plot_pil_list(images_list: list[Image.Image], width: int = 10, length: int = 16) -> None:
    for i, image in enumerate(images_list):
        plt.figure(figsize=(width, length))
        plt.imshow(image)
        plt.axis("off")
        plt.title(str(i))
        plt.show()


def has_keywords(filename: str, keywords: Iterable[str]) -> bool:
    clean_filename = unidecode(filename).lower()
    return any(keyword.lower() in clean_filename for keyword in keywords)


def walk_search(paths: Iterable[str], keywords: Iterable[str]) -> dict[str, dict[str, dict[str, str]]]:
    path_dict = {}

    for folder in paths:
        folder_dict = {}
        for (dir_path, dir_names, filenames) in os.walk(folder):
            file_dict = {}
            for filename in filenames:
                if has_keywords(filename, keywords):
                    file_dict[filename] = os.path.join(dir_path, filename)
            folder_dict[dir_path] = file_dict
        path_dict[folder] = folder_dict

    return path_dict


def write_search(path_dict: dict[str, dict[str, dict[str, str]]], filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(path_dict, f, ensure_ascii=False, indent=4, sort_keys=True)


def rotate_pdf(filename: str, rotations: dict[int, int]) -> str:
    reader = PdfReader(filename)
    writer = PdfWriter()

    for i in range(len(reader.pages)):
        writer.add_page(reader.pages[i])

    for page, degrees in rotations.items():
        writer.pages[page-1].rotate((360 - degrees) % 360)  # degree change to be consistent with PIL

    with open(filename, "wb") as f:
        writer.write(f)

    return filename


# noinspection PyUnresolvedReferences
def merge_pdfs(paths: list[str], filename: str) -> str:
    result = fitz.open()

    for pdf in paths:
        with fitz.open(pdf) as f:
            result.insert_pdf(f)

    _save_name(filename, "merged")
    result.save(filename)
    return filename


def correcting_orientations(folder: str) -> list:
    return cnn_eval.correcting_orientations(folder)


def wrong_orientations(folder: str) -> list[str]:
    wrong = list()
    for file, correcting in correcting_orientations(folder):
        if correcting != 0:
            wrong.append(file)
    return wrong


