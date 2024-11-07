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
from test import *


def train_model(
        model, 
        trainloader, 
        testloader,
        criterion, 
        optimizer, 
        scheduler, 
        num_epochs=10, 
        starting_epoch=0, 
        device="cuda",
        seed=42):
    
    for epoch in range(starting_epoch, starting_epoch+num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        t1 = time.time()
        for inputs, labels in tqdm(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        t2 = time.time()

        per_epoch_time = t2-t1
        print(f"Took {per_epoch_time} seconds/epoch")

        # Compute average training loss
        avg_train_loss = running_loss / len(trainloader)

        # Compute validation loss and accuracy
        val_loss, val_accuracy = evaluate_model_loss(model, testloader, criterion, "cuda")

        # Step the scheduler based on validation loss
        scheduler.step(avg_train_loss)


        checkpoint_path = f"./resnet_epoch_{epoch+1}.pth"
        # checkpoint_path = os.path.join(checkpoint_dir, f"resnet_epoch_{epoch+1}.pth")

        torch.save({
            'seed': seed,
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),  # Save the scheduler state
            'train_loss': avg_train_loss,
            'per_epoch_time': per_epoch_time
        }, checkpoint_path)

        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')
    return model

