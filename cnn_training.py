import sys
import os
from timeit import default_timer as timer
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from fich import _save_name as get_name
import cnn_model

# Arguments: learning rate, transformation (1-3), reduction factor

device = cnn_model.device

print(f"Using {device if device != 'cuda' else 'ROCm'} device")

# Path to labeled database
input_folder = os.path.join("fich_database", "inputs_scanned")

# Define data augmentation transforms
transform_train = {
    '1': [transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(400, ratio=(0.4, 1.0), antialias=True),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=15)
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
print(f"Loading data from {input_folder}...")
dataset = datasets.ImageFolder(input_folder, transform_train[sys.argv[2]][0])
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
print(model)
loss_fn = cnn_model.loss_fn
cnn_model.set_optimizer(learning_rate)
optimizer = cnn_model.optimizer
# Define the learning rate scheduler
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.01, patience=2, verbose=True)

accuracies = list()
start = timer()
print("Training started...")

for t in range(num_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    cnn_model.train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler)
    correct = cnn_model.test_loop(dataloader, model, loss_fn)
    accuracies.append(correct)

finish = timer()
print("Done!")
execution_time = finish - start

print("Saving model and optimizer states...")

if not os.path.exists("training_logs/"):
    os.makedirs("training_logs/")

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

print("Finished.")
