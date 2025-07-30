import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import models.resnet as resnet
import modules.data_loaders as data_loader
import mat_mul
import argparse 
import sys
import os
trained_models_path = "./trained_models/"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

def calibration(model, stats=False):
    print("CALIBRATING")
    if stats:
        model.eval()
    else:
        model.train()
    #for batch, (inputs, _) in enumerate(tqdm(train_loader)):
    for batch, (inputs, _) in enumerate(train_loader):
        inputs = inputs.to(device)
        model(inputs)

def train(epoch, model, optimizer, criterion, conv_type=2):
    model.train()
    """first_conv_layer = model.layer1.conv1
    # Questi attributi devono essere stati impostati all'interno della classe Conv2d_custom
    # durante la calibrazione/training, come nel tuo codice originale.
    print(f"  Activation Scale: {first_conv_layer.activation_scale}")
    print(f"  Activation ZP Neg: {first_conv_layer.activation_zp_neg}")
    print(f"  Weight Scale: {first_conv_layer.weight_scale}")
    print(f"  Weight ZP Neg: {first_conv_layer.weight_zp_neg}")"""
    total_loss, correct, total = 0, 0, 0
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
    print("TESTING")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        #for inputs, targets in tqdm(test_loader):
        for inputs, targets in (test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1) 
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print(f"Test Accuracy: {100.*correct/total:.2f}%")
    return (100.*correct/total)

train_loader, test_loader, _ = data_loader.get_datasets(64)
device = 'cuda'

def new_training_method(multiplier_matrix=None, pretrained=False, retrain=False, conv_type=1, bit_width=8, signed=False, epochs=5):
    print(f"Network training with parameters: multiplier_matrix = {multiplier_matrix} conv_type = {conv_type}, bit_width = {bit_width}, signed = {signed}")
    
    models_dir = trained_models_path.rstrip('/') 

    if not os.path.exists(models_dir):
        print(f"The model folder '{models_dir}' does not exist. Creating it now...")
        try:
            os.makedirs(models_dir)
            print(f"Folder '{models_dir}' created successfully.")
        except OSError as e:
            print(f"Error creating folder '{models_dir}': {e}")
            return 
        
    best_accuracy = 0
    if not pretrained:
        model = resnet.ResNet8(multiplier_matrix, num_classes=10, conv_type=1, bit_width=bit_width, signed=signed).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 135], gamma=0.1)

        for epoch in range(200):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            train(epoch, model, optimizer, criterion)
            scheduler.step()
        test(model)
        torch.save(model.state_dict(), trained_models_path + 'resnet.pth')
        del model

    if conv_type == 1:
        try:
            print("Loading pretrained model")
            model = resnet.ResNet8(multiplier_matrix, num_classes=10, conv_type=1, bit_width=bit_width, signed=signed).to(device)
            model.load_state_dict(torch.load(trained_models_path + 'resnet.pth', weights_only=True))
            test(model)
            return
        except Exception as e:
            raise RuntimeError(f"No pretrained model found for conv_type 1: {e}")

    model = resnet.ResNet8(multiplier_matrix, num_classes=10, conv_type=conv_type, bit_width=bit_width, signed=signed).to(device)
    if conv_type != 1:
        try:
            if conv_type == 2:
                model.load_state_dict(torch.load(trained_models_path + 'resnet.pth', weights_only=True))
            else:
                model.load_state_dict(torch.load(trained_models_path + 'resnet_q' + str(bit_width) + '.pth', weights_only=True))
        except Exception as e: 
            raise RuntimeError(f"No pretrained model found for conv_type {conv_type}: {e}")

        calibration(model)
        if conv_type == 5:
            calibration(model, stats=True)
            return

        if retrain:
            best_accuracy = 0
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)
            start_accuracy = test(model)
            for epoch in range(epochs):
                print(f"Epoch {epoch + 1}\n-------------------------------")
                train(epoch, model, optimizer, criterion, conv_type)
                scheduler.step()
                acc = test(model)
                if(acc > best_accuracy):
                    best_accuracy = acc
            if conv_type == 2:
                torch.save(model.state_dict(), trained_models_path + 'resnet_q' + str(bit_width) + '.pth')
        else:
            start_accuracy = test(model)
        return start_accuracy if start_accuracy is not None else 0.0, best_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training methods with specified parameters."
    )
    
    # Optional positional argument for input_path
    parser.add_argument(
        "--input_path",
        nargs='?', # Makes it optional
        default=None,
        help="Path to an .npy file or a directory containing .npy files for training."
    )
    
    # Optional arguments for training parameters
    parser.add_argument(
        "--conv_type",
        type=int,
        default=1, # Default value as in your initial no-arg case
        help="Convolution type for the training method (default: 1)."
    )
    parser.add_argument(
        "--bit_width",
        type=int,
        default=8, # Default value
        help="Bit width for the training method (default: 8)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3, # Default value as in your initial no-arg case
        help="Number of training epochs (default: 3)."
    )
    parser.add_argument(
        "--retrain",
        action="store_true", # This makes it a boolean flag
        default = True,
        help="If set, retrain the model. Otherwise, use existing weights."
    )
    parser.add_argument(
        "--pretrained",
        action="store_true", # This makes it a boolean flag
        default = True,
        help="If set, load a pretrained model. Otherwise, train from scratch."
    )
    parser.add_argument(
        "--signed",
        action="store_true", # This makes it a boolean flag
        default = False,
        help="If set, inputs are treated as signed integers. Default is unsigned."
    )

    args = parser.parse_args()

    # Scenario 1: No input_path provided (default behavior from original script)
    if args.input_path is None:
        print("No input path provided. Running default training method.")
        new_training_method(
            None, 
            pretrained=args.pretrained, 
            retrain=args.retrain, 
            conv_type=args.conv_type, 
            bit_width=args.bit_width, 
            signed=args.signed, 
            epochs=args.epochs
        )
        sys.exit(0)

    # Scenario 2: input_path provided (file or directory)
    input_path = args.input_path

    if not os.path.exists(input_path):
        print(f"Error: The input path '{input_path}' does not exist. Please create it and place .npy files inside.")
        sys.exit(1) # Use sys.exit(1) for error exit codes

    if os.path.isfile(input_path):
        print(f"Processing single file: {input_path}")
        result = new_training_method(
            input_path, 
            pretrained=args.pretrained, 
            retrain=args.retrain, 
            conv_type=args.conv_type, 
            bit_width=args.bit_width, 
            signed=args.signed, 
            epochs=args.epochs
        )
        print(result)
    elif os.path.isdir(input_path):
        print(f"Processing directory: {input_path}")
        best_acc_list = {}
        for filename in os.listdir(input_path):
            if filename.endswith('.npy'):
                input_full_path = os.path.join(input_path, filename)
                print(f"  Training for {filename}...")
                best_acc_list[filename] = new_training_method(
                    input_full_path, 
                    pretrained=args.pretrained, 
                    retrain=args.retrain, 
                    conv_type=args.conv_type, 
                    bit_width=args.bit_width, 
                    signed=args.signed, 
                    epochs=args.epochs
                )
        print("\nTraining results for directory:")
        print(best_acc_list)
