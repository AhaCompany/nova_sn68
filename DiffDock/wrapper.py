# -*- coding: utf-8 -*-

import os
import pandas as pd
import torch
import math
import json

from torch_geometric.data import DataLoader

from .diffdock_utils.dataset import ProteinLigandDataset
from .diffdock_utils.data_utils import virtual_screening
from .diffdock_utils import protein_init, ligand_init
from .models.network import DiffDockModel

from .runtime_config import RuntimeConfig

class DiffDockWrapper:
    def __init__(self):
        self.runtime_config = RuntimeConfig()
        self.device = self.runtime_config.DEVICE
        
        # Check if CUDA is available and update device if needed
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            print(f"CUDA requested but not available, defaulting to CPU")
            self.device = 'cpu'
            self.runtime_config.DEVICE = 'cpu'
        
        # Initialize the model
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        """
        Initialize the DiffDock model with parameters from config
        and load pretrained weights if available
        """
        # Load config from file
        config_path = os.path.join(self.runtime_config.MODEL_PATH, 'config.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            model_params = config.get('model_params', {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load config file: {e}")
            print("Using default model parameters")
            model_params = {
                "node_features": 16,
                "edge_features": 4,
                "hidden_dim": 128,
                "depth": 4,
                "time_emb_dim": 32,
                "heads": 4,
                "dropout": 0.1
            }
        
        # Create model with parameters from config
        model = DiffDockModel(
            node_features=model_params.get('node_features', 16),
            edge_features=model_params.get('edge_features', 4),
            hidden_dim=model_params.get('hidden_dim', 128),
            depth=model_params.get('depth', 4),
            time_emb_dim=model_params.get('time_emb_dim', 32),
            heads=model_params.get('heads', 4),
            dropout=model_params.get('dropout', 0.1)
        ).to(self.device)
        
        # Try to load pretrained weights
        weights_path = os.path.join(self.runtime_config.MODEL_PATH, 'model.pt')
        try:
            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path, map_location=self.device))
                print(f"Successfully loaded model weights from {weights_path}")
            else:
                print(f"Warning: No pretrained weights found at {weights_path}")
                print("Using randomly initialized weights. Results may be suboptimal.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Using randomly initialized weights. Results may be suboptimal.")
        
        return model
    
    def initialize_protein(self, protein_seq:str) -> dict:
        """
        Initialize protein from sequence
        """
        self.protein_seq = [protein_seq]
        protein_dict = protein_init(self.protein_seq)
        return protein_dict
    
    def initialize_smiles(self, smiles_list:list) -> dict:
        """
        Initialize ligands from SMILES strings
        """
        self.smiles_list = smiles_list
        smiles_dict = ligand_init(smiles_list)
        return smiles_dict
    
    def create_screen_loader(self, protein_dict, smiles_dict):
        """
        Create a DataLoader for screening
        """
        self.screen_df = pd.DataFrame({
            'Protein': [k for k in protein_dict for _ in smiles_dict],
            'Ligand': [l for l in smiles_dict for _ in protein_dict],
        })
        
        dataset = ProteinLigandDataset(
            self.screen_df,
            smiles_dict,
            protein_dict,
            device=self.device
        )
        
        self.screen_loader = DataLoader(
            dataset,
            batch_size=self.runtime_config.BATCH_SIZE,
            shuffle=False
        )
        
    def run_challenge_start(self, protein_seq:str):
        """
        Start a new challenge with a protein sequence
        """
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self.protein_dict = self.initialize_protein(protein_seq)
        self.model.eval()  # Set model to evaluation mode
        
    def run_validation(self, smiles_list:list) -> pd.DataFrame:
        """
        Run validation on a list of SMILES strings
        
        Args:
            smiles_list: List of SMILES strings to validate
            
        Returns:
            DataFrame with predicted binding affinities
        """
        self.smiles_dict = self.initialize_smiles(smiles_list)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        self.create_screen_loader(self.protein_dict, self.smiles_dict)
        
        self.screen_df = virtual_screening(
            self.screen_df,
            self.model,
            self.screen_loader,
            os.getcwd(),
            save_interpret=False,
            ligand_dict=self.smiles_dict,
            device=self.device,
            save_cluster=False
        )
        
        return self.screen_df