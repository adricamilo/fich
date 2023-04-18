import os
import tempfile
import pdf2image
import fich
import gc

pdf_folder = os.path.join("fich_database", "pdfs")
file_set = set()

for dir_, _, files in os.walk(pdf_folder):
    for file_name in files:
        file = os.path.join(pdf_folder, file_name)
        if os.path.isfile(file) and file.endswith(".pdf"):
            file_set.add(file)

image_folder = os.path.join("fich_database", "images")

counter = 0

for file in file_set:
    with tempfile.TemporaryDirectory() as path:
        images = pdf2image.convert_from_path(file, output_folder=path)
        for image in images:
            # noinspection PyProtectedMember
            image.save(fich._get_name(counter, "page", ".png"))
            counter = counter + 1
    print(file, gc.collect())
