import torch.nn
import torch.optim


class CNNModel(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(CNNModel, self).__init__()

        # Convolutional layers
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3),  # 3 input channels, 32 output channels, 3x3 kernel
            torch.nn.ReLU(),  # ReLU activation
            torch.nn.BatchNorm2d(32),  # Batch normalization after convolution
            torch.nn.MaxPool2d(2, stride=2),  # Max pooling with 2x2 kernel and stride 2
            torch.nn.Dropout2d(p=0.25),  # Dropout with drop probability of 0.25

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
            torch.nn.Linear(512, 3),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.dense_layers(x)


# Create an instance of the CNNModel class
model = CNNModel()
criterion = torch.nn.CrossEntropyLoss()  # Use CrossEntropyLoss as the loss function
optimizer = None  # set optimizer as None until set_optimizer is called


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
def set_optimizer(lr: float = 0.001, mt: float = 0.9):
    global optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=mt)
