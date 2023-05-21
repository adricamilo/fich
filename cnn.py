import torch.nn
import torch.optim
import numpy
from sklearn.metrics import confusion_matrix
import pandas
from pandas.core.frame import DataFrame
from torch.utils.data.dataloader import DataLoader


class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolutional layers
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3),          # 3 input channels, 32 output channels, 3x3 kernel
            torch.nn.ReLU(),                    # ReLU activation
            torch.nn.BatchNorm2d(32),           # Batch normalization after convolution
            torch.nn.MaxPool2d(2, stride=2),    # Max pooling with 2x2 kernel and stride 2
            torch.nn.Dropout2d(p=0.25),         # Dropout with drop probability of 0.25

            torch.nn.Conv2d(32, 64, 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Dropout2d(p=0.25),

            torch.nn.Conv2d(64, 128, 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Dropout2d(p=0.25),

            torch.nn.Conv2d(128, 256, 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Dropout2d(p=0.25),

            torch.nn.Conv2d(256, 512, 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Dropout2d(p=0.25)
        )

        # Dense layers
        self.dense_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(512 * 100, 512),  # 100 factor comes from conv layers and 400×400 input image
            torch.nn.ReLU(),
            torch.nn.Linear(512, 3)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.dense_layers(x)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Create an instance of the CNNModel class
model = CNNModel()
loss_fn = torch.nn.CrossEntropyLoss()  # Use CrossEntropyLoss as the loss function
optimizer = None  # set optimizer as None until set_optimizer is called
lr_scheduler = None  # set lr_scheduler as None until set_optimizer is called

# Load previous model state dictionary
# model.load_state_dict(torch.load("training_logs/..."))

model.to(device)


def callable_once(func):
    def wrapper(*args, **kwargs):
        if wrapper.called:
            raise ValueError(f"{func.__name__} function can only be called once")
        else:
            wrapper.called = True
            return func(*args, **kwargs)

    wrapper.called = False
    return wrapper


@callable_once
def set_optimizer(lr: float = 0.01):
    global optimizer, lr_scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer.load_state_dict(torch.load("training_logs/..."))
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2, verbose=True)


def train_loop(dataloader: DataLoader):
    global model, loss_fn, optimizer, lr_scheduler
    model.train()
    # noinspection PyTypeChecker
    size = len(dataloader.dataset)
    correct = 0
    validate = 0
    for batch, (X, y) in enumerate(dataloader):
        # GPU optimization
        images, labels = X.to(device), y.to(device)

        # Compute prediction and loss
        outputs = model(images)
        pred = torch.argmax(outputs, dim=1)
        correct += (pred == labels).type(torch.FloatTensor).sum().item()
        loss = loss_fn(outputs, labels)
        validate += loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 25 == 0:
            loss, current = loss.item(), (batch + 1) * len(images)
            print(f"loss: {loss:>7f}  [{current:>4d}/{size:>4d}]")

    correct /= size
    print(f"Error in train set: \n Accuracy: {(100 * correct):>0.1f}%")

    validate /= size
    lr_scheduler.step(validate)


def test_loop(dataloader: DataLoader) -> float:
    global model, loss_fn
    model.eval()
    # noinspection PyTypeChecker
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            images, labels = X.to(device), y.to(device)
            outputs = model(images)
            pred = torch.argmax(outputs, dim=1)
            test_loss += loss_fn(outputs, labels).item()
            correct += (pred == labels).type(torch.FloatTensor).sum().item()

    test_loss /= num_batches
    correct /= size
    print(type(correct))
    print(f"Error in test set: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return correct


def test_loop_conf_matrix(dataloader: DataLoader) -> tuple[float, DataFrame]:
    global model
    model.eval()

    # noinspection PyTypeChecker
    size = len(dataloader.dataset)
    correct = 0
    y_pred = list()
    y_true = list()

    with torch.no_grad():
        for X, y in dataloader:
            images, labels = X.to(device), y.to(device)
            outputs = model(images)
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == labels).type(torch.FloatTensor).sum().item()
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    correct /= size

    classes = ("clockwise", "counterclockwise", "upright")

    print("Building confusion matrix...")
    conf_matrix = confusion_matrix(y_true, y_pred)
    df_conf_matrix = pandas.DataFrame(conf_matrix / numpy.sum(conf_matrix, axis=1)[:, None], index=[i for i in classes],
                                      columns=[i for i in classes])
    print(type(df_conf_matrix))
    return correct, df_conf_matrix
