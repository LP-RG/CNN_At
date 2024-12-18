import torch
import os
import numpy as np
from torch.autograd import Function
from torch import nn
import torch.ao.quantization
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import custom_op
import time

errors_matrix = torch.from_numpy(np.load('res_diff_matrix8192.npy')).float().to("cuda")
derivative_matrix_y = torch.from_numpy(np.load('derivative_matrix_y.npy')).float().to("cuda")
derivative_matrix_x = torch.from_numpy(np.load('derivative_matrix_x.npy')).float().to("cuda")

def quantize(input:torch.Tensor,def_scale):
    input_clamped = torch.clamp(input,min = -1, max = 6)
    scale = 7/float(def_scale)
    quantized_tensor = torch.round(input_clamped/scale)
    return quantized_tensor, scale


####### NETWORK DEFINITION #######
def gradient_error_inputs(input:torch.Tensor,kernel:torch.Tensor, grad_ouput:torch.Tensor,stride,padding):
    batch_size, in_channels, start_height, start_width = input.size()
    out_channels, _, kernel_height, kernel_width = kernel.size()    
    input_unfolded = nn.functional.unfold(input,kernel_size=(kernel_height, kernel_width),stride=stride).transpose(1,2)
    kernel_flatten = kernel.view(out_channels, -1).T
    output = torch.ops.custom_op.derivate(input_unfolded.contiguous(),kernel_flatten.contiguous(),derivative_matrix_x.contiguous(),grad_ouput.contiguous()).contiguous()
    output = torch.sum(output, dim=1,keepdim=False)
    output = output.transpose(1,2).contiguous()
    output = nn.functional.fold(output, output_size=(start_height - (2 * padding[0]) , start_width - (2 * padding[1])), kernel_size=(kernel_height, kernel_width), stride=stride, padding= padding)
    return output

def gradient_error_weights(input:torch.Tensor,kernel:torch.Tensor, grad_ouput:torch.Tensor,stride):
    batch_size, in_channels, _, _ = input.size()
    out_channels, _, kernel_height, kernel_width = kernel.size()    
    input_unfolded = nn.functional.unfold(input,kernel_size=(kernel_height, kernel_width),stride=stride).transpose(1,2)
    kernel_flatten = kernel.view(out_channels, -1).T
    output = torch.ops.custom_op.derivate(input_unfolded.contiguous(),kernel_flatten.contiguous(),derivative_matrix_y.contiguous(),grad_ouput.contiguous()).contiguous()
    output = torch.sum(output, dim=0,keepdim=False)
    output = output/batch_size
    output = torch.sum(output, dim=1,keepdim=False)
    output = output.view(out_channels,in_channels,kernel_height,kernel_width)
    return output

def convolution(input:torch.Tensor,kernel:torch.Tensor,bias: torch.Tensor,stride,act_scale,filter_scale): 
    batch_size, _, in_height, in_width = input.size()
    out_channels, _, kernel_height, kernel_width = kernel.size()    
    input_unfolded = nn.functional.unfold(input,kernel_size=(kernel_height, kernel_width),stride=stride)    
    kernel_flatten = kernel.view(out_channels, -1)  
    output =  torch.ops.custom_op.convolution(input_unfolded.transpose(1,2).contiguous(),kernel_flatten.T.contiguous(),errors_matrix.contiguous(),act_scale,filter_scale)
    #output =  torch.ops.custom_op.convolution_no_error(input_unfolded.transpose(1,2).contiguous(),kernel_flatten.T.contiguous(),act_scale,filter_scale)
    output = output.transpose(1,2) 
    output_height = (in_height - kernel_height) // stride[0] + 1
    output_width = (in_width - kernel_width) // stride[1] + 1
    output = output.view(batch_size, out_channels, output_height, output_width) 
    output_biased = output  
    if(bias != None):
        output_biased=output+bias.view(1,out_channels,1,1)    
    return output_biased 
    
class Conv2d_custom(nn.Conv2d):
    def __init__(self,channel_in,channel_out,kernel_size,stride,padding):
        super().__init__(channel_in,channel_out,kernel_size,stride,padding)
    def forward(self, input):
        return CustomConv2d.apply(input,self.weight, self.bias, self.stride, self.padding) 
    
class CustomConv2d(Function): 
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding):
        # Salviamo i tensori necessari per il calcolo del backward
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        m=nn.ZeroPad2d((padding[1],padding[1],padding[0],padding[0]))
        input_padded=m(input) 
        quantized_act, act_scale = quantize(input_padded,255)
        quantized_filter, filter_scale = quantize(weight,255)
        ctx.act_scale = act_scale
        ctx.filter_scale = filter_scale
        # Eseguiamo la convoluzione (utilizzeremo la funzione nn.Conv2d per semplicità)
        return convolution(quantized_act,quantized_filter,bias,stride,act_scale,filter_scale) 
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        m = nn.ZeroPad2d((padding[1], padding[1], padding[0], padding[0]))
        input_padded = m(input)
        quantized_act, _ = quantize(input_padded, 255)
        quantized_filter, _ = quantize(weight, 255)
        
        # Calcolo ottimizzato
        error_derivate_weights = gradient_error_weights(
            quantized_act.contiguous(), quantized_filter.contiguous(), grad_output, stride
        )
        grad_weight = torch.nn.grad.conv2d_weight(
            input, weight.shape, grad_output, stride, padding
        ) - error_derivate_weights
        
        # Libera i tensor non necessari
        del error_derivate_weights
        
        error_derivate_inputs = gradient_error_inputs(
            quantized_act.contiguous(), quantized_filter.contiguous(), grad_output, stride, padding
        )
        grad_input = torch.nn.grad.conv2d_input(
            input.shape, weight, grad_output, stride, padding
        ) - error_derivate_inputs
        
        # Libera i tensor non necessari
        del error_derivate_inputs, quantized_act, quantized_filter, input_padded
        torch.cuda.empty_cache()  # Svuota esplicitamente la cache della GPU
        
        grad_bias = grad_output.sum((0, 2, 3))
        return grad_input, grad_weight, grad_bias, None, None

 
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatter=nn.Flatten()
        self.conv_layers1=nn.Sequential(
            #nn.Conv2d(1, 8, kernel_size=3,stride=1,padding=1),
            Conv2d_custom(1, 8, 3,(1,1),(1,1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            #nn.Conv2d(8, 16, kernel_size=3,stride=1,padding=1),
            Conv2d_custom(8, 16, 3,(1,1),(1,1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            #nn.Conv2d(16, 32, kernel_size=3,stride=1,padding=1),
            Conv2d_custom(16, 32, 3,(1,1),(1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            Conv2d_custom(32, 64, 3,(1,1),(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            Conv2d_custom(64, 32, 3,(1,1),(1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU()         
        )
        self.dense_layers=nn.Sequential(
            nn.Linear(25088 ,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,10)            
        )
    def forward(self,x):
        x=self.conv_layers1(x)
        x=self.flatter(x)
        x=self.dense_layers(x)
        return x


######## TEST AND TRAINING LOOP #########      
training_data=datasets.FashionMNIST(root="data",
                                    train=True,
                                    download=True,
                                    transform=ToTensor())
test_data=datasets.FashionMNIST(root="data",
                                train=False,
                                download=True,
                                transform=ToTensor())


batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size,shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size,shuffle=True)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device=("cuda")
print(f"Using {device} device")

model=CNNModel()
#model.load_state_dict(torch.load('starting_weights.pth', weights_only=True))
model = model.to(device)
torch.save(model.state_dict(), 'starting_weights_big.pth')
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size=len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        X,y=X.to(device), y.to(device)
        pred=model(X)
        loss=loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if(batch % 25 == 0):
            loss, current = loss.item(), (batch + 1) * len(X)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]",file= open("accur_train_corrected_big.txt","a"))

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n",file = open("accur_test_corrected_big.txt","a"))

epochs = 30
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)    
    test(test_dataloader, model, loss_fn)
print("Done!")

