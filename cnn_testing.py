import os
from timeit import default_timer as timer
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cnn
import matplotlib.pyplot as plt
import seaborn as sn

print(f"Using {cnn.device if cnn.device != 'cuda' else 'cuda/ROCm'} device")

# Path to testing database
input_folder = os.path.join("fich_database", "inputs_testing")

# Define data augmentation transforms
transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((400, 400), antialias=True)
    ])

print(f"Loading data from {input_folder}...")

dataset_test = datasets.ImageFolder(input_folder, transform_test)
# noinspection PyUnresolvedReferences
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=32, shuffle=False)

start = timer()
print("Testing started...")

correct, df_conf_matrix = cnn.test_loop_conf_matrix(dataloader_test)

finish = timer()
print("Done!")

execution_time = finish - start

minutes = int(execution_time // 60)
seconds = execution_time % 60

print(f"Accuracy: {(100 * correct):>0.1f}% \nTime to test: {minutes} m {seconds:>0.01f} s")

plt.title("dropout scanned (with photos train set)")
sn.heatmap(df_conf_matrix, annot=True, vmin=0, vmax=1)

plt.show()
# plt.savefig("training_logs/best/confusion matrices/...")
