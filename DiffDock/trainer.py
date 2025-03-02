# -*- coding: utf-8 -*-

import os
import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from DiffDock.diffdock_utils.dataset import ProteinLigandDataset
from DiffDock.diffdock_utils import protein_init, ligand_init
from DiffDock.models.network import DiffDockModel
from DiffDock.runtime_config import RuntimeConfig

def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def save_config(config, config_path):
    """Save configuration to JSON file"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def setup_model(config, device):
    """Set up DiffDock model based on config parameters"""
    model_params = config['model_params']
    
    model = DiffDockModel(
        node_features=model_params.get('node_features', 16),
        edge_features=model_params.get('edge_features', 4),
        hidden_dim=model_params.get('hidden_dim', 128),
        depth=model_params.get('depth', 4),
        time_emb_dim=model_params.get('time_emb_dim', 32),
        heads=model_params.get('heads', 4),
        dropout=model_params.get('dropout', 0.1)
    ).to(device)
    
    return model

def load_dataset(data_path, protein_dict, ligand_dict, device):
    """Load and prepare dataset for training"""
    df = pd.read_csv(data_path)
    
    # Ensure the dataset has the required columns
    required_columns = ['Protein', 'Ligand', 'binding_affinity']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Dataset missing required column: {col}")
    
    dataset = ProteinLigandDataset(df, ligand_dict, protein_dict, device=device)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    return train_dataset, val_dataset

def train_epoch(model, dataloader, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(batch)
        
        # Get ground truth binding affinities
        target = batch.binding_affinity.to(device)
        
        # Compute loss
        loss = F.mse_loss(pred, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def validate(model, dataloader, device):
    """Validate the model"""
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Forward pass
            pred = model(batch)
            
            # Get ground truth binding affinities
            target = batch.binding_affinity.to(device)
            
            # Compute loss
            loss = F.mse_loss(pred, target)
            val_loss += loss.item()
    
    return val_loss / len(dataloader)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train DiffDock model')
    parser.add_argument('--data', type=str, required=True, help='Path to training data CSV')
    parser.add_argument('--config', type=str, default='DiffDock/trained_weights/config.json', help='Path to config file')
    parser.add_argument('--output', type=str, default='DiffDock/trained_weights', help='Output directory for model weights')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'
    print(f"Using device: {device}")
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"Loaded configuration from {args.config}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading config: {e}")
        print("Using default configuration")
        config = {
            "model_params": {
                "node_features": 16,
                "edge_features": 4,
                "hidden_dim": 128,
                "depth": 4,
                "time_emb_dim": 32,
                "heads": 4,
                "dropout": 0.1
            },
            "diffusion_params": {
                "num_timesteps": 1000,
                "beta_schedule": "cosine",
                "beta_start": 1e-4,
                "beta_end": 0.02
            },
            "training_params": {
                "batch_size": 32,
                "learning_rate": 1e-4,
                "weight_decay": 1e-6,
                "epochs": 100
            },
            "inference_params": {
                "sampling_steps": 20,
                "temperature": 1.0,
                "top_k": 5
            }
        }
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Save config to output directory
    output_config_path = os.path.join(args.output, 'config.json')
    save_config(config, output_config_path)
    
    # Load data
    print(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    
    # Initialize protein and ligand dictionaries
    protein_sequences = df['protein_sequence'].unique().tolist()
    protein_dict = protein_init(protein_sequences)
    
    smiles_list = df['smiles'].unique().tolist()
    ligand_dict = ligand_init(smiles_list)
    
    # Prepare datasets
    train_dataset, val_dataset = load_dataset(args.data, protein_dict, ligand_dict, device)
    
    # Create dataloaders
    batch_size = config['training_params'].get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Set up model
    model = setup_model(config, device)
    
    # Set up optimizer
    lr = config['training_params'].get('learning_rate', 1e-4)
    weight_decay = config['training_params'].get('weight_decay', 1e-6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop
    epochs = config['training_params'].get('epochs', 100)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.output, 'model.pt')
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(args.output, 'training_curve.png'))
    
    print("Training complete!")

if __name__ == "__main__":
    main()