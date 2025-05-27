import torch
import numpy as np
#from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import models.resnet as resnet
import modules.data_loaders as data_loader
import mat_mul
import time

trained_models_path = "./trained_models/"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

def calibration(model, stats = False):
    print("CALIBRATING")
    if(stats):
        model.eval()
    else:
        model.train()
    #for batch, (inputs, _) in (enumerate(tqdm(train_loader))):
    for batch, (inputs, _) in (enumerate(train_loader)):
        inputs = inputs.to(device)
        model(inputs)

# Training loop 
def train(epoch,model,optimizer,criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        if(batch % 100 == 0):
            print(f"loss: {loss:>7f}  [{batch:>5d}/{len(train_loader):>5d}]")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print(f"Epoch {epoch + 1}: Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100.*correct/total:.2f}%")


# Testing loop
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


train_loader,test_loader, _ = data_loader.get_datasets(64)
device = 'cuda'

def new_training_method(pretrained = False, retrain = False, conv_type = 1, bit_width = 8, signed = False):
    print(f"Network training with parameters: conv_type = {conv_type}, bit_width = {bit_width}, signed = {signed}")
    if(not pretrained):
        model = resnet.ResNet8(num_classes=10,conv_type=1,bit_width=bit_width,signed=signed).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
                model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
            )
        scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[90, 135], gamma=0.1
            )
        # Run training and testing
        for epoch in range(200): 
            print(f"Epoch {epoch + 1}\n-------------------------------")
            train(epoch,model,optimizer,criterion)
            scheduler.step()
        test(model)
        torch.save(model.state_dict(), trained_models_path + 'resnet.pth')
        del model
    if(conv_type == 1):
        try:
            print("Loading pretrained model")	
            model = resnet.ResNet8(num_classes=10,conv_type=1,bit_width=bit_width,signed=signed).to(device)
            model.load_state_dict(torch.load(trained_models_path + 'resnet.pth', weights_only=True))
            test(model)
            return
        except:
            raise RuntimeError("No pretrained model found")
    model = resnet.ResNet8(num_classes=10,conv_type=conv_type,bit_width=bit_width,signed=signed).to(device)
    if(conv_type != 1):
        try:
            if(conv_type == 2):
                model.load_state_dict(torch.load(trained_models_path + 'resnet.pth', weights_only=True))
            else:
                model.load_state_dict(torch.load(trained_models_path + 'resnet_q'+str(bit_width) + '.pth', weights_only=True))
        except:
            raise RuntimeError("No pretrained model found")
        
        calibration(model)
        if(conv_type == 5):
                #TODO DELETE EXISTING FILES
                calibration(model, stats=True)
                return

        test(model)
        if(retrain):
            best_accuracy = 0
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                    model.parameters(), lr=0.0001
                )
            scheduler = optim.lr_scheduler.StepLR(optimizer= optimizer, step_size=10, gamma = 0.5)
            for epoch in range(3):
                print(f"Epoch {epoch + 1}\n-------------------------------")
                train(epoch, model, optimizer, criterion)
                current_accuracy = test(model)
                if(current_accuracy > best_accuracy):
                    best_accuracy = current_accuracy
                scheduler.step()

            if(conv_type == 2):
                torch.save(model.state_dict(), trained_models_path + 'resnet_q'+str(bit_width) + '.pth')

            #test(model)
        return best_accuracy

#new_training_method(pretrained=True, retrain=True, conv_type=3, bit_width=4, signed=False)
