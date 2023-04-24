import sys
import os
from timeit import default_timer as timer
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import tempfile
import shutil
from fich import _save_name as get_name
import cnn_model

# noinspection PyUnresolvedReferences
dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=32, shuffle=False)


# Testeo
model.eval()
print("Testing...")
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in dataloader_test:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        # noinspection PyUnresolvedReferences
        correct += (predicted == labels).sum().item()
    print("Test accuracy: {:.2f}%".format(correct / total * 100))
