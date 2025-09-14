import torch
import numpy as np
import mat_mul
import torch.nn as nn
import torch.optim as optim
import models.resnet20 as resnet20
import models.lenet5 as lenet5
import models.vgg16 as vgg16
import models.alexnet_cifar10 as alexnet_cifar10
import models.resnet56 as resnet56
import modules.data_loaders as data_loader
import argparse
import sys
import os
import time
trained_models_path = "./trained_models/"
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

device = "cuda"
MODEL_FACTORIES = {
    "resnet": resnet20.ResNet20,
    "lenet5": lenet5.LeNet5,
    "vgg16": vgg16.VGG16,
    "alexnet_cifar10": alexnet_cifar10.AlexNetCIFAR10,
    "resnet56": resnet56.ResNet56_CIFAR100,
}
train_loader = None
test_loader = None
_classes = None

def set_data_loaders(model_name: str, batch_size: int = 64):
    global train_loader, test_loader, _classes
    name = model_name.lower()

    if name in ("lenet5", "resnet"):
        batch_size = 64
    elif name == "vgg16":
        batch_size = 128
    elif name == "alexnet_cifar10":
        batch_size = 128
    elif name == "resnet56":
        batch_size = 128

    train_loader, test_loader, _classes = data_loader.get_datasets(batch_size, model_name)

def get_exact_training_setup(model_name: str, model: nn.Module):
    name = model_name.lower()

    if name == "resnet":
        epochs = 100
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
        return epochs, optimizer, scheduler

    if name == "lenet5":
        epochs = 20
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return epochs, optimizer, scheduler

    if name == "vgg16":
        epochs = 100
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, weight_decay=0.005, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return epochs, optimizer, scheduler

    if name == "alexnet_cifar10":
        epochs = 200
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return epochs, optimizer, scheduler

    if name == "resnet56":
        epochs = 200
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return epochs, optimizer, scheduler

    epochs = 100
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    return epochs, optimizer, scheduler

def calibration(model, stats=False):
    print("Calibrating model...")
    if stats:
        model.eval()
    else:
        model.train()
    for inputs, _ in train_loader:
        inputs = inputs.to(device)
        model(inputs)

def train_one_epoch(epoch, model, optimizer, criterion):
    print(f"Training epoch {epoch + 1}...")
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        if batch % 100 == 0:
            print(f"loss: {loss:>7f}  [{batch:>5d}/{len(train_loader):>5d}]")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print(f"Epoch {epoch + 1}: Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100.*correct/total:.2f}%")

def test(model):
    print("Testing model...")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        acc = 100.0 * correct / total
        print(f"Test Accuracy: {acc:.2f}%")
    return acc

def build_model(model_name: str, conv_type: int, bit_width: int, signed: bool, zone: bool, multiplier_matrix=None, num_classes: int = 10):
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Model '{model_name}' non supportato.")
    return MODEL_FACTORIES[model_name](
        multiplier_matrix,
        num_classes=num_classes,
        conv_type=conv_type,
        bit_width=bit_width,
        signed=signed,
        zone=zone
    ).to(device)

def new_training_method(model_name: str, multiplier_matrix=None, conv_type: int = 1, bit_width: int = 8, signed: bool = False, zone: bool = False, exact_accuracy: float = 0):
    print(f"Network training with parameters: model_name = {model_name}, conv_type = {conv_type}, bit_width = {bit_width}, signed = {signed}, zone = {zone}")
    models_dir = trained_models_path.rstrip('/')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    exact_path = os.path.join(models_dir, f"{model_name}.pth")
    quant_path = os.path.join(models_dir, f"{model_name}_q{bit_width}.pth")
    num_classes = _classes if _classes else 10
    print(num_classes)
    if conv_type == 1:
        if os.path.exists(exact_path):
            print("Carico modello esatto e avvio test...")
            model = build_model(model_name, conv_type=1, bit_width=bit_width, signed=signed, zone=zone, multiplier_matrix=multiplier_matrix, num_classes=num_classes)
            model.load_state_dict(torch.load(exact_path, weights_only=True))
            calibration(model)
            return test(model)
        else:
            print("Alleno modello esatto...")
            model = build_model(model_name, conv_type=1, bit_width=bit_width, signed=signed, zone=zone, multiplier_matrix=multiplier_matrix, num_classes=num_classes)
            epochs, optimizer, scheduler = get_exact_training_setup(model_name, model)
            criterion = nn.CrossEntropyLoss()
            for epoch in range(epochs):
                print(f"Epoch {epoch + 1}\n-------------------------------")
                train_one_epoch(epoch, model, optimizer, criterion)
                scheduler.step()
            torch.save(model.state_dict(), exact_path)
            return test(model)

    if conv_type == 2:
        exact_exists = os.path.exists(exact_path)
        quant_exists = os.path.exists(quant_path)
        if exact_exists and (not quant_exists):
            print("Training quantizzato (5 epoche)...")
            model = build_model(model_name, conv_type=2, bit_width=bit_width, signed=signed, zone=zone, multiplier_matrix=multiplier_matrix, num_classes=num_classes)
            model.load_state_dict(torch.load(exact_path, weights_only=True), strict=False)
            calibration(model)
            criterion = nn.CrossEntropyLoss()
            lr = 0.001 if bit_width == 4 else 0.0001
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)
            best_acc = 0.0
            for epoch in range(5):
                print(f"Epoch {epoch + 1}\n-------------------------------")
                train_one_epoch(epoch, model, optimizer, criterion)
                scheduler.step()
                acc = test(model)
                best_acc = max(best_acc, acc)
            torch.save(model.state_dict(), quant_path)
            return best_acc
        if exact_exists and quant_exists:
            print("Test modello quantizzato...")
            model = build_model(model_name, conv_type=2, bit_width=bit_width, signed=signed, zone=zone, multiplier_matrix=multiplier_matrix, num_classes=num_classes)
            model.load_state_dict(torch.load(quant_path, weights_only=True))
            calibration(model)
            return test(model)
        raise RuntimeError("Allena prima il modello esatto.")

    if conv_type == 3:
        if not os.path.exists(quant_path):
            raise RuntimeError("Allena prima il modello quantizzato.")
        print("Retrain modello approssimato (3 epoche)...")
        model = build_model(model_name, conv_type=3, bit_width=bit_width, signed=signed, zone=zone, multiplier_matrix=multiplier_matrix, num_classes=num_classes)
        model.load_state_dict(torch.load(quant_path, weights_only=True))
        calibration(model)
        criterion = nn.CrossEntropyLoss()
        lr = 0.001 if bit_width == 4 else 0.0001
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)
        best_accuracy = 0
        for epoch in range(3):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            train_one_epoch(epoch, model, optimizer, criterion)
            scheduler.step()
            acc = test(model)
            if(acc < exact_accuracy - 2):
                return 0.0
            if acc > best_accuracy:
                best_accuracy = acc
        return best_accuracy

    raise ValueError(f"conv_type={conv_type} non supportato.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training with simplified logic and model_name routing.")
    parser.add_argument("--model_name", type=str, default="resnet")
    parser.add_argument("--conv_type", type=int, default=1)
    parser.add_argument("--bit_width", type=int, default=8)
    parser.add_argument("--signed", action="store_true", default=False)
    parser.add_argument("--zone", action="store_true", default=False)
    parser.add_argument("--input_path", nargs='?', default=None)
    parser.add_argument("--exact_accuracy", type=float, default=0)
    args = parser.parse_args()
    set_data_loaders(args.model_name)
    start = time.time()
    p = args.input_path
    if p is None:
        print(new_training_method(args.model_name, None, args.conv_type, args.bit_width, args.signed, args.zone,args.exact_accuracy))
        sys.exit(0)

    if not os.path.exists(p):
        print(f"Error: The input path '{p}' does not exist.")
        sys.exit(1)

    if os.path.isfile(p):
        print(new_training_method(args.model_name, p, args.conv_type, args.bit_width, args.signed, args.zone,args.exact_accuracy))
    else:
        results = {
            f: new_training_method(args.model_name, os.path.join(p, f), args.conv_type, args.bit_width, args.signed, args.zone,args.exact_accuracy)
            for f in os.listdir(p) if f.endswith(".npy")
        }
        print(results)
    print(f"Total training time: {time.time() - start}")
