import os
import fitz
import fich

pdf_folder = os.path.join("fich_database", "pdfs")
file_list = list()

for dir_, _, files in os.walk(pdf_folder):
    for file_name in files:
        file = os.path.join(pdf_folder, file_name)
        if os.path.isfile(file) and file.endswith(".pdf"):
            file_list.append(file)

file_list.sort()

print(len(file_list))

image_folder = os.path.join("fich_database", "images")

page_num = 0
file_num = 1

for file in file_list:
    print(f"Starting file {file_num}/{len(file_list)}...", file)
    # noinspection PyUnresolvedReferences
    with fitz.open(file) as pdf:
        for page in pdf:
            pix = page.get_pixmap()  # render page to an image
            # noinspection PyProtectedMember
            name = os.path.join(image_folder, fich._get_name(page_num, "page", ".png"))
            pix.save(name)  # store image as a PNG
            page_num = page_num + 1
    print(f"Completed file {file_num}/{len(file_list)}")
    file_num = file_num + 1
