import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from train import *
from test import *



def set_seed(seed = 42):
    # Set seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Setting seed of {seed}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for pre-trained models
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# dataset that is to be used : CIFAR 10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# training, testing and validation set
# train_size = int(0.8 * len(trainset))
# val_size = len(trainset) - train_size
# trainset, valset = random_split(trainset, [train_size, val_size])


# dataloader for loading dataset into the network
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
# valloader = DataLoader(valset, batch_size=64, shuffle=False)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

#deciding the model
model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # Using SGD with momentum

scheduler = ReduceLROnPlateau(optimizer, mode='min')



load_model = False


if load_model ==  True:
    checkpoint = torch.load("./resnet_epoch_9.pth",map_location=torch.device("cuda")) ## use the name of your checkpoint file
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    starting_epoch = checkpoint['epoch']
    seed = checkpoint['seed']
    set_seed(seed)
    num_epochs = 10
    start_time = time.time()
    trained_model = train_model(model, trainloader,testloader, criterion, optimizer, scheduler, num_epochs=num_epochs, starting_epoch=0, seed=seed)
    end_time = time.time()
    print(f"It took around {end_time - start_time} seconds for training model to {num_epochs} epochs")

else:
    seed = 42
    set_seed(seed)
    starting_epoch = 0
    num_epochs = 10
    start_time = time.time()
    trained_model = train_model(model, trainloader,testloader, criterion, optimizer, scheduler, num_epochs=num_epochs, starting_epoch=0, seed=seed)
    end_time = time.time()
    print(f"It took around {end_time - start_time} seconds for training model to {num_epochs} epochs")
                              

test_loss, test_accuracy = evaluate_model_loss(model, testloader, criterion)
print(f'\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')


    