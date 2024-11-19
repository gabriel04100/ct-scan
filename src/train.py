import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
from datetime import datetime
import mlflow
import mlflow.pytorch
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Module
from typing import Any, Optional
import torchvision.models as models
import torch.nn.functional as F
from torch.optim import Adam
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
from itertools import product
import torchvision
from torch.optim import Optimizer
import torch.optim as optim


# Transformation personnalisée pour ajouter du bruit gaussien
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),  # Toujours convertir en tenseur en dernier
    AddGaussianNoise(mean=0.0, std=0.05)
])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Convertir les images en tenseurs
])


def initialize_model(num_classes: int = 4) -> Module:
    """
    Initialize a ResNet-18 model and adapt it for a specified number o
    of output classes.

    Parameters:
    ----------
    num_classes : int, optional
        The number of output classes for the model. Default is 4.

    Returns:
    -------
    torch.nn.Module
        The modified ResNet-18 model with a custom fully connected layer.
    """
    model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_model(
    model: Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    criterion: Module,
    optimizer: Optimizer,
    num_epochs: int = 10,
    patience: int = 5
) -> Module:
    """
    Train a model using a given dataset, with early stopping 
    and MLflow integration.

    Parameters:
    ----------
    model : torch.nn.Module
        The neural network model to be trained.
    train_loader : torch.utils.data.DataLoader
        DataLoader providing batches of training data.
    valid_loader : torch.utils.data.DataLoader
        DataLoader providing batches of validation data.
    criterion : torch.nn.Module
        Loss function used for optimization.
    optimizer : torch.optim.Optimizer
        Optimizer used for updating model weights.
    num_epochs : int, optional
        Maximum number of training epochs. Default is 10.
    patience : int, optional
        Number of epochs to wait for an improvement in validation accuracy
        before stopping early. Default is 5.

    Returns:
    -------
    torch.nn.Module
        The trained model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_accuracy = 0.0
    epochs_no_improve = 0
    early_stop = False

    with mlflow.start_run() as run:  # Start an MLflow run
        mlflow.log_params({
            "learning_rate": optimizer.defaults['lr'],
            "weight_decay": optimizer.defaults['weight_decay'],
            "num_epochs": num_epochs,
            "patience": patience
        })

        for epoch in range(num_epochs):
            if early_stop:
                break
     
            model.train()
            running_loss = 0.0
    
            for inputs, labels in tqdm(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            accuracy = evaluate_model(model, valid_loader)
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f},'
                  'Accuracy: {accuracy:.2f}%')

            # Log metrics to MLflow
            mlflow.log_metrics({
                "epoch_loss": epoch_loss,
                "epoch_accuracy": accuracy
            }, step=epoch)
            
            # Save the model if it's the best
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                path_format = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                model_path = f'../models/best_model_resnet18_{path_format}.pth'
                torch.save(model.state_dict(), model_path)
                mlflow.log_artifact(model_path)  # Log the best model to MLflow
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print('Early stopping triggered')
                    early_stop = True
        
        # Log the best accuracy at the end of the run
        mlflow.log_metric("best_accuracy", best_accuracy)

    return model


def evaluate_model(model: Module, dataloader: DataLoader) -> float:
    """
    Evaluate the accuracy of a given model on a dataset.

    Parameters:
    ----------
    model : torch.nn.Module
        The model to be evaluated.
    dataloader : torch.utils.data.DataLoader
    Returns:
    -------
    float
        The accuracy of the model on the dataset, as a percentage.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            corrects += torch.sum(preds == labels.data).item()

    accuracy = 100 * corrects / total
    return accuracy


if __name__ == "__main__":

    train_dataset = datasets.ImageFolder(root="../data/train",
                                         transform=train_transform)
    test_dataset = datasets.ImageFolder(root="../data/test",
                                        transform=transform)
    valid_dataset = datasets.ImageFolder(root="../data/valid",
                                         transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=10, shuffle=True)

    mlflow.set_tracking_uri('../mlruns')

    # Define the model, criterion, and optimizer
    model = initialize_model(num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Run the training function
    num_epochs = 50
    trained_model = train_model(model, train_loader, valid_loader,
                                criterion, optimizer, num_epochs=num_epochs)
