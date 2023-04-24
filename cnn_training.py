import sys
import os
from timeit import default_timer as timer
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import tempfile
import shutil
from fich import _save_name as get_name
import cnn_model

# Arguments: learning rate, transformation (1-3), reduction factor

# Path to labeled database
data_dir = os.path.join("fich_database", "inputs_scanned")

print("Creating temporary directory for training...")
tmp_dir = tempfile.TemporaryDirectory()
tmp_datos = os.path.join(tmp_dir.name, "datos")
print(f"Created {tmp_datos}...")
print(f"Copying data from {data_dir}...")
shutil.copytree(data_dir, tmp_datos)

# Define data augmentation transforms
transform_train = {
    '1': [transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(400, ratio=(0.4, 1.0), antialias=True),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=15)
    ]), "RandomResizedCrop, RandomAffine"],
    '2': [transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(400)
    ]), "CenterCrop"],
    '3': [transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((400, 400), antialias=True)
    ]), "Resize"]
}
print("Loading data...")
dataset = datasets.ImageFolder(tmp_datos, transform_train[sys.argv[2]][0])
N = len(dataset)
print(f"Full training dataset size: {N}")
factor = int(sys.argv[3])
indices = range(0, N, factor)
# noinspection PyUnresolvedReferences
dataset_train = torch.utils.data.Subset(dataset, torch.tensor(indices))
print(f"Training dataset size reduced by a factor of {factor}")
# noinspection PyUnresolvedReferences
dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True)

# Hyperparameters
learning_rate = float(sys.argv[1])
batch_size = 32
num_epochs = 10

# Model, loss function and optimizer
model = cnn_model.model
criterion = cnn_model.criterion
cnn_model.set_optimizer(learning_rate)
optimizer = cnn_model.optimizer
# Define the learning rate scheduler
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2, verbose=True)

start = timer()
accuracies = []
model.train()
print("Training started...")

for epoch in range(num_epochs):
    # len(dataloader) = numero de archivos / batch_size
    N = len(dataloader) // 3
    for i, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print training statistics
        if (i + 1) % N == 0:
            # Calcular la precisión de entrenamiento
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            # noinspection PyUnresolvedReferences
            correct = (predicted == labels).sum().item()
            accuracy = correct / total
            batch_log = "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%".format(
                epoch + 1,num_epochs, i + 1, len(dataloader), loss.item(), accuracy * 100)
            print(batch_log)
            accuracies.append(batch_log)
    # noinspection PyUnboundLocalVariable
    lr_scheduler.step(loss)  # Update the learning rate


finish = timer()
print("Done!")
execution_time = finish - start

print("Saving model and optimizer states...")

model_name = get_name("training_logs", "model", ".pt")
optim_name = get_name("training_logs", "optim", ".pt")

torch.save(model.state_dict(), model_name)
torch.save(optimizer.state_dict(), optim_name)

print("Saving logs...")

# Logger file
log_file = open(get_name("training_logs", "logs", ".txt"), "w")
log_file.write(f"learning rate = {learning_rate}" + "\n")
log_file.write(f"transformation = {transform_train[sys.argv[2]][1]}" + "\n")
log_file.write(f"training set reduction factor = {factor}" + "\n")
log_file.write(f"time to train = {execution_time // 60} m" + "\n")
for acc in accuracies:
    log_file.write(acc + "\n")
log_file.close()

print("Cleaning up...")
tmp_dir.cleanup()
