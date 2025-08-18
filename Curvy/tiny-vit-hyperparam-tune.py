#!/usr/bin/env python3
"""
ResNet-18 Hyperparameter Tuning with Curvy Optimizer
W&B Sweeps integration for automated hyperparameter optimization
"""

import os
import argparse
import json
from datetime import datetime
from itertools import product
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import wandb
import matplotlib.pyplot as plt

from simplevit import SimpleViT
from PyHessian.pyhessian.hessian import hessian
from curvy import Curvy

# ================================
# Configuration and Constants
# ================================

# Experiment configuration
EXPERIMENT_CONFIG = {
    'project_name': 'simple-vit-hessian-project-curvy',
    'entity': None,  # Set to your W&B username/team if needed
    'num_epochs': 20,
    'dataset': 'Imagenette',
    'model': 'TinyViT',
    'device': 'cuda:1' if torch.cuda.is_available() else 'cpu'
}

# Default hyperparameters (used when not running sweeps)
DEFAULT_HYPERPARAMETERS = {
    'batch_size': 64,
    'learning_rate': 1e-2,
    'hessian_epsilon': 1e-2,
    'momentum': 0.9,
    'hessian_compute_interval': 50,
    'hessian_n_iter': 100
}

# Data transforms
TRANSFORMS = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}


# ================================
# Data Loading Functions
# ================================

def preload_dataset(dataset):
    images = []
    labels = []
    for img, label in dataset:
        images.append(img)
        labels.append(label)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return torch.utils.data.TensorDataset(images, labels)

def load_datasets():
    """Load Imagenette training and test datasets"""
    train_dataset = datasets.Imagenette(
        root='./data', 
        split='train', 
        transform=TRANSFORMS['train'], 
        download=True
    )
    test_dataset = datasets.Imagenette(
        root='./data', 
        split='val', 
        transform=TRANSFORMS['test'], 
        download=True
    )
    return preload_dataset(train_dataset), preload_dataset(test_dataset)


def create_data_loaders(train_dataset, test_dataset, batch_size):
    """Create data loaders for training and testing"""
    g = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=g
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        generator=g
    )
    return train_loader, test_loader


# ================================
# Model and Training Functions
# ================================

def train_vit_sweep():
    """
    Train TinyViT model with W&B sweep hyperparameters
    This function is called by W&B sweep agents
    """
    # Initialize W&B run (sweep will provide hyperparameters)
    wandb.init()
    
    # Get hyperparameters from W&B config
    config = wandb.config
    
    # Set device
    device = torch.device(EXPERIMENT_CONFIG['device'])
    print(f"Using device: {device}")
    print(f"Hyperparameters: {dict(config)}")
    
    # Load datasets
    train_dataset, test_dataset = load_datasets()
    train_x = torch.stack([x[0] for x in train_dataset])
    train_y = torch.tensor([x[1] for x in train_dataset])
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        train_dataset, 
        test_dataset, 
        config.batch_size
    )

    # Initialize TinyViT model
    model = SimpleViT(image_size=224, patch_size=16, num_classes=10)
    model = model.to(device)
    
    # Log model info to W&B
    wandb.config.update({
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    })
    
    # Initialize Curvy optimizer
    optimizer = Curvy(
        model.parameters(),
        model=model,
        lr=config.learning_rate,
        hessian_epsilon=config.hessian_epsilon,
        momentum=config.momentum,
        hessian_compute_interval=config.hessian_compute_interval,
        hessian_n_iter=config.hessian_n_iter,
        criterion=nn.CrossEntropyLoss(),
        train_x=train_x,
        train_y=train_y,
        hessian_computer=hessian,
        cuda=True,
        cuda_device=device
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training tracking
    num_epochs = EXPERIMENT_CONFIG['num_epochs']
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # ================
        # Training Phase
        # ================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        train_loss /= len(train_loader)
        train_accuracy = 100.0 * train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # ================
        # Testing Phase
        # ================
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        
        test_loss /= len(test_loader)
        test_accuracy = 100.0 * test_correct / test_total
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # Print progress
        print(f'Epoch {epoch+1:2d}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
        
        # Log to W&B
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'learning_rate': config.learning_rate
        })
    
    # Log final metrics for sweep optimization
    final_metrics = {
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
        'final_train_accuracy': train_accuracies[-1],
        'final_test_accuracy': test_accuracies[-1],
        'best_train_accuracy': max(train_accuracies),
        'best_test_accuracy': max(test_accuracies),
        'min_train_loss': min(train_losses),
        'min_test_loss': min(test_losses)
    }
    
    wandb.log(final_metrics)
    wandb.summary.update(final_metrics)
    
    return model, train_losses, test_losses, train_accuracies, test_accuracies


def plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies):
    """Plot training and testing curves"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, test_losses, 'r-', label='Testing Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Testing Loss Curves')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    ax2.plot(epochs, test_accuracies, 'r-', label='Testing Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Testing Accuracy Curves')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


# ================================
# Sweep and Manual Training Functions
# ================================

def train_single_run(hyperparams=None):
    """
    Train a single run with specified hyperparameters (for manual/testing)
    """
    if hyperparams is None:
        hyperparams = DEFAULT_HYPERPARAMETERS
    
    # Initialize W&B run
    wandb.init(
        project=EXPERIMENT_CONFIG['project_name'],
        entity=EXPERIMENT_CONFIG['entity'],
        config=hyperparams
    )
    
    # Set device
    device = torch.device(EXPERIMENT_CONFIG['device'])
    print(f"Using device: {device}")
    print(f"Hyperparameters: {hyperparams}")
    
    # Load datasets
    train_dataset, test_dataset = load_datasets()
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        train_dataset, 
        test_dataset, 
        hyperparams['batch_size']
    )

    # Initialize ResNet-18 model
    model = models.resnet18(pretrained=False, num_classes=10)
    model = model.to(device)
    
    # Initialize Curvy optimizer
    optimizer = Curvy(
        model.parameters(),
        model=model,
        lr=hyperparams['learning_rate'],
        hessian_epsilon=hyperparams['hessian_epsilon'],
        momentum=hyperparams['momentum'],
        hessian_compute_interval=hyperparams['hessian_compute_interval'],
        hessian_n_iter=hyperparams['hessian_n_iter'],
        criterion=nn.CrossEntropyLoss(),
        hessian_computer=hessian,
        cuda=True
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = EXPERIMENT_CONFIG['num_epochs']
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        train_loss /= len(train_loader)
        train_accuracy = 100.0 * train_correct / train_total
        
        # Testing Phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        
        test_loss /= len(test_loader)
        test_accuracy = 100.0 * test_correct / test_total

        # Print progress
        print(f'Epoch {epoch+1:2d}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
        
        # Log to W&B
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'learning_rate': hyperparams['learning_rate']
        })
    
    # Log final metrics
    wandb.log({
        'final_train_loss': train_loss,
        'final_test_loss': test_loss,
        'final_train_accuracy': train_accuracy,
        'final_test_accuracy': test_accuracy
    })
    
    wandb.finish()
    return model


# ================================
# Sweep Utilities
# ================================

def create_sweep():
    """Create a W&B sweep and return the sweep ID"""
    sweep_config = {
        'method': 'bayes',  # or 'grid', 'random'
        'metric': {
            'goal': 'minimize',
            'name': 'final_test_loss'
        },
        'parameters': {
            'batch_size': {
                'values': [32, 64, 128]
            },
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-1
            },
            'hessian_epsilon': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-1
            },
            'momentum': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 0.99
            },
            'hessian_compute_interval': {
                'values': [10, 50, 100, 200]
            },
            'hessian_n_iter': {
                'value': 100
            }
        }
    }
    
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=EXPERIMENT_CONFIG['project_name'],
        entity=EXPERIMENT_CONFIG['entity']
    )
    
    return sweep_id


# ================================
# Main Execution
# ================================

def main():
    """Main function to handle different execution modes"""
    parser = argparse.ArgumentParser(description='TinyVit Curvy Optimizer Training')
    parser.add_argument('--mode', choices=['sweep', 'agent', 'single'], default='single',
                        help='Execution mode: sweep (create sweep), agent (run sweep agent), single (single run)')
    parser.add_argument('--sweep_id', type=str, help='Sweep ID for agent mode')
    parser.add_argument('--count', type=int, default=200, help='Number of runs for sweep agent')
    
    args = parser.parse_args()
    
    if args.mode == 'sweep':
        # Create a new sweep
        print("Creating W&B sweep...")
        sweep_id = create_sweep()
        print(f"Sweep created! Sweep ID: {sweep_id}")
        print(f"Run sweep agents with: python {__file__} --mode agent --sweep_id {sweep_id}")
        
    elif args.mode == 'agent':
        # Run sweep agent
        if not args.sweep_id:
            print("Error: --sweep_id required for agent mode")
            return
        
        print(f"Starting sweep agent for sweep: {args.sweep_id}")
        print(f"Will run up to {args.count} experiments")
        
        wandb.agent(
            sweep_id=args.sweep_id,
            function=train_vit_sweep,
            count=args.count,
            project=EXPERIMENT_CONFIG['project_name'],
            entity=EXPERIMENT_CONFIG['entity']
        )
        
    else:
        # Single run mode
        print("Running single experiment with default hyperparameters...")
        train_single_run()


if __name__ == "__main__":
    main()