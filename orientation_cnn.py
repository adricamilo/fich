import os.path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import tempfile
import shutil
import fich

data_dir = os.path.join("fich_database", "inputs")

print("Creating temporary directory...")
tmp_dir = tempfile.TemporaryDirectory()
tmp_datos = os.path.join(tmp_dir.name, "datos")
print(f"Temporary directory with data: {tmp_datos}")
print(f"Copying data from {data_dir}...")
shutil.copytree(data_dir, tmp_datos)
print("Copied data")

transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((400, 400), antialias=True)
])
# Solamente lee la estructura en clases de la base de datos
g_cpu = torch.Generator()
g_cpu.manual_seed(8230147051205205078)
# noinspection PyUnresolvedReferences
dataset_train, dataset_test = torch.utils.data.random_split(
    datasets.ImageFolder(tmp_datos, transformation),
    [8/10, 2/10],
    g_cpu
)
# Se puede decirle que cambie el tamaño de las imágenes
# Solamente es un procedimiento para lectura de archivos, las imágenes no se han cargado
# noinspection PyUnresolvedReferences
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True)
# noinspection PyUnresolvedReferences
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=32, shuffle=False)


# Definir la red neuronal convolucional
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Preprocesamiento de imágenes
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.25)
        )

        # Red per se
        self.linear_layers = nn.Sequential(
            nn.Linear(512 * 100, 512),
            nn.Linear(512, 3),
            nn.Softmax(dim=1)
        )

        # Flattening
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.linear_layers(x)
        return x


# Definir los hiperparámetros
learning_rate = 0.001
batch_size = 32
num_epochs = 10

# Instanciar el modelo y definir la función de pérdida y el optimizador
model = Net()
# Criterio de clasificación
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Modelo original usa RMSprop

accuracies = []
model.train()
print("Training started...")
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(dataloader_train):

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calcular la precisión de entrenamiento
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / total

        # Imprimir estadísticas de entrenamiento
        if (i + 1) % 39 == 0:
            # len(dataloader) = numero de archivos / batch_size
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, len(dataloader_train), loss.item(), accuracy * 100))
            accuracies.append(accuracy)

print("Training finished")

print("Saving model and optimizer states...")

model_name = fich._save_name("fich_database", "modelADAM", ".pt")
optim_name = fich._save_name("fich_database", "optimADAM", ".pt")

torch.save(model.state_dict(), model_name)
torch.save(optimizer.state_dict(), optim_name)
print(f"Saved to {model_name} and {optim_name}!")

# plt.ylim([0, 1])
# plt.xlabel("Time")
# plt.ylabel("Accuracy")
# plt.plot(accuracies)

log_name = fich._save_name("fich_database", "training_accuraciesADAM", ".txt")
print(f"Writing training accuracy logs to {log_name}...")
with open(log_name, "w") as f:
    for acc in accuracies:
        f.write(str(acc) + "\n")
print("Done!")

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
        correct += (predicted == labels).sum().item()
    print("Test accuracy: {:.2f}%".format(correct / total * 100))

print("Cleaning up...")
tmp_dir.cleanup()

print("Finished!")
