import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import logging
import os

torch.manual_seed(220403)


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv2d(1, 6, 5)
        self.s2 = nn.MaxPool2d(2, 2)
        self.c3 = nn.Conv2d(6, 16, 5)
        self.s4 = nn.MaxPool2d(2, 2)
        self.c5 = nn.Conv2d(16, 120, 5)
        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = self.s2(x)
        x = F.relu(self.c3(x))
        x = self.s4(x)
        x = F.relu(self.c5(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.f6(x))
        x = self.f7(x)

        return x


class QLenet5(LeNet5):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = super().forward(x)
        x = self.dequant(x)

        return x


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
    ]
)

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)

test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transform, download=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)


def train(model, criterion, optimizer, loader, device=torch.device("cpu")):
    model.train()

    total_loss = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, metric, device=torch.device("cpu")):
    model.eval()
    metric.reset()

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            metric(outputs, targets)

    return metric.compute().item()


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

EPOCHS = 5
MODEL_PATH = "tmp/lenet5.pt"
QUANTIZED_MODEL_PATH = "tmp/lenet5_int8.pt"

if __name__ == "__main__":
    if not os.path.isfile(MODEL_PATH):
        logging.info("Model not found")

        logging.info("Started training")
        model = LeNet5().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        acc_metric = torchmetrics.Accuracy("multiclass", num_classes=10).to(device)

        for epoch in range(EPOCHS):
            train_loss = train(model, criterion, optimizer, train_loader, device)
            accuracy = evaluate(model, test_loader, acc_metric, device)

            logging.info(
                f"Epoch: {epoch + 1}/{EPOCHS}, "
                f"Loss: {train_loss:.2f}, "
                f"Accuracy: {accuracy:.2f}"
            )
        logging.info("Ended training")
        torch.jit.script(model).save("tmp/lenet5.pt")
        logging.info("Saved model")
    if not os.path.isfile(QUANTIZED_MODEL_PATH):
        logging.info("Quantized model is not found")

        model = torch.jit.load(MODEL_PATH).to("cpu")
        acc_metric = torchmetrics.Accuracy("multiclass", num_classes=10)

        accuracy = evaluate(model, test_loader, acc_metric)
        logging.info(f"Accuracy before quantization: {accuracy:.2f}")

        logging.info("Started quantization")
        model_int8 = QLenet5()
        model_int8.load_state_dict(model.state_dict())
        model_int8.eval()
        model_int8.qconfig = torch.quantization.get_default_qconfig("qnnpack")
        torch.quantization.prepare(model_int8, inplace=True)
        logging.info("Started calibration")
        with torch.no_grad():
            for inputs, outputs in train_loader:
                inputs = inputs
                model_int8(inputs)
        logging.info("Finished calibration")
        torch.quantization.convert(model_int8, inplace=True)
        logging.info("Finished quantization")

        acc_metric.reset()
        accuracy = evaluate(model_int8, test_loader, acc_metric, device)

        logging.info(f"Accuracy after quantization: {accuracy:.2f}")

        torch.jit.script(model_int8).save(QUANTIZED_MODEL_PATH)
        logging.info("Saved quantized model")
