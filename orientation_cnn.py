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

data_dir = "datos"

tmp_dir = tempfile.TemporaryDirectory()
tmp_datos = os.path.join(tmp_dir.name, "datos")
print(tmp_datos)
shutil.copytree(data_dir, tmp_datos)

transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256,256))
])
# Solamente lee la estructura en clases de la base de datos
g_cpu = torch.Generator()
g_cpu.manual_seed(8230147051205205078)
dataset_train, dataset_test = torch.utils.data.random_split(
    datasets.ImageFolder(tmp_datos, transformation),
    [8/10, 2/10],
    g_cpu
)
# Se puede decirle que cambie el tamaño de las imágenes
# Solamente es un procedimiento para lectura de archivos, las imágenes no se han cargado
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=32, shuffle=False)


# Definir la red neuronal convolucional
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Preprocesamiento de imágenes
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Red per se
        self.linear_layers = nn.Sequential(
            nn.Linear(16 * 62 * 62, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # Flatten
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

# Definir los hiperparámetros
learning_rate = 0.001
batch_size = 32
num_epochs = 8

# Instanciar el modelo y definir la función de pérdida y el optimizador
model = Net()
# Criterio de clasificación
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

accuracies = []
model.train()
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
        if (i + 1) % 100 == 0:
            # len(dataloader) = numero de archivos / batch_size
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, len(dataloader_train), loss.item(), accuracy * 100))
            accuracies.append(accuracy)

plt.ylim([0, 1])
plt.xlabel("Time")
plt.ylabel("Accuracy")
plt.plot(accuracies)

# Testeo
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in dataloader_test:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test accuracy: {:.2f}%'.format(correct / total * 100))

tmp_dir.cleanup()