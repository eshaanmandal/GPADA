import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import random
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

# setting  the seed
seed  = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # If using GPU
torch.cuda.manual_seed_all(seed)  # For multi-GPU environments
np.random.seed(seed)
random.seed(seed)


batch_size = 16
num_classes = 10
device = "cuda" if torch.cuda.is_available() else "cpu"



#setting the transforms for CIFAR-10
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),   # Random crop with padding
    transforms.RandomHorizontalFlip(),      # Randomly flip horizontally
    transforms.ToTensor(),                  # Convert to tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10(
    root='../../data',
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = datasets.CIFAR10(
    root='../../data',
    train=False,
    download=True,
    transform=test_transform
)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


#define the predictor model
class PredictorModel(nn.Module):
    def __init__(self, output_size=4096):
        super(PredictorModel, self).__init__()
        
        # Define layers
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, output_size)
    
    def forward(self, x):
        # Reshape if input is a flat vector
        # print(x.shape)
        if len(x.shape) == 2:
            batch_size, features = x.shape
            side_length = int(features/2)
            x = x.view(batch_size, 1, side_length, 2)

        elif len(x.shape) == 4:
            x = x.mean(dim=0, keepdim=True)
            x = x.permute(1, 0, 2, 3)
        
        # Standardize input size using adaptive pooling or interpolation
        if x.size(2) < 7 or x.size(3) < 7:
            x = F.interpolate(x, size=(7, 7), mode='bilinear', align_corners=False)
        else:
            print(f"x shape before adaptive_pool: {x.shape}")
            print(f"x device: {x.device}")
            x = self.adaptive_pool(x)
            print(f"x shape after adaptive_pool: {x.shape}")
        
        # Pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        # Get the fixed-size output
        return self.fc(x)
    

class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, predicted, target):
        epsilon = 1e-8  # Small value to avoid division by zero
        return torch.mean(torch.abs((target - predicted) / (target + epsilon))) 


#lets define the model we want to train
main_model = models.resnet50(weights=None)
main_model.fc = nn.Linear(main_model.fc.in_features, num_classes)
main_model = main_model.to(device)

predictor_model = PredictorModel(5000).to(device)
mape_loss = MAPELoss()
criterion = nn.CrossEntropyLoss()

main_optimizer = optim.Adam(main_model.parameters(), lr=1e-4)
predictor_optimizer = optim.SGD(predictor_model.parameters(), lr=0.01, momentum=0.9)

activations = {}
def hook_fn(module, input, output):
    activations[module] = output.detach()  # Detach to prevent tracking in the computational graph

for name, layer in main_model.named_modules():
    if isinstance(layer, (nn.Conv2d, nn.Linear)):  # Attach hooks to Conv2d and Linear layers
        layer.register_forward_hook(hook_fn)


num_epochs = 10
for epoch in range(num_epochs):
    main_model.train()
    predictor_model.train()

    epoch_main_loss = 0.0  # For tracking main model's training loss
    epoch_mape_loss = 0.0  # For tracking predictor model's MAPE loss
    num_batches = 0
    
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass through the main model (hooks will capture activations)
        outputs = main_model(inputs)
        loss_main = criterion(outputs, labels) # Dummy loss for gradient calculation

        # Backward pass to get gradients in the main model
        main_optimizer.zero_grad()
        loss_main.backward()
        main_optimizer.step()

        epoch_main_loss += loss_main.item()
        
        # Loop through each layer's activation and gradient to train predictor model
        total_mape_loss = 0.0
        for layer, activation in activations.items():
            if hasattr(layer, 'weight') and layer.weight.grad is not None:
                # Get the actual gradient for weights
                actual_weight_grad = layer.weight.grad.detach().view(-1)
                target_size = actual_weight_grad.size(0)
                
                # Predict a fixed-size output and interpolate to target size
                fixed_predicted_gradients = predictor_model(activation)
                print("got the predicted gradients, now lets interpolate")
                interpolated_predicted_gradients = F.interpolate(
                    fixed_predicted_gradients.view(1, 1, -1), 
                    size=(target_size,),
                    mode='linear', 
                    align_corners=False
                ).view(-1)
                
                # Calculate MAPE loss for weights
                layer_mape_loss = mape_loss(interpolated_predicted_gradients, actual_weight_grad)
                total_mape_loss += layer_mape_loss

                # Optionally: Handle bias gradients if the layer has a bias
                if layer.bias is not None and layer.bias.grad is not None:
                    actual_bias_grad = layer.bias.grad.detach().view(-1)
                    bias_target_size = actual_bias_grad.size(0)
                    
                    # Interpolate for bias gradients
                    bias_interpolated_predicted_gradients = F.interpolate(
                        fixed_predicted_gradients.view(1, 1, -1),
                        size=(bias_target_size,),
                        mode='linear',
                        align_corners=False
                    ).view(-1)
                    
                    # Compute MAPE loss for biases
                    bias_mape_loss = mape_loss(bias_interpolated_predicted_gradients, actual_bias_grad)
                    total_mape_loss += bias_mape_loss

        # Update predictor model based on total MAPE loss for all layers
        predictor_optimizer.zero_grad()
        total_mape_loss.backward()
        predictor_optimizer.step()

        epoch_mape_loss += total_mape_loss.item()
        num_batches += 1
    
    avg_main_loss = epoch_main_loss / num_batches
    avg_mape_loss = epoch_mape_loss / num_batches
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Cross Entropy Loss: {avg_main_loss:.4f}, Average MAPE Loss: {avg_mape_loss:.4f}")




