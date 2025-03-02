import torch
from torch_geometric.data import Data, Dataset
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import os
from Bio.PDB import PDBParser
import torch.nn.functional as F

class ProteinLigandDataset(Dataset):
    """
    Dataset for protein-ligand binding prediction
    """
    def __init__(self, df, ligand_dict, protein_dict, device='cuda:0'):
        super(ProteinLigandDataset, self).__init__()
        self.df = df
        self.ligand_dict = ligand_dict
        self.protein_dict = protein_dict
        self.device = device
        
    def len(self):
        return len(self.df)
    
    def get(self, idx):
        """
        Get a single protein-ligand pair
        """
        row = self.df.iloc[idx]
        protein_id = row['Protein']
        ligand_id = row['Ligand']
        
        # Get protein and ligand data
        protein_data = self.protein_dict[protein_id]
        ligand_data = self.ligand_dict[ligand_id]
        
        # Create a data object
        data = self._create_graph(protein_data, ligand_data, idx)
        
        return data
    
    def _create_graph(self, protein_data, ligand_data, idx):
        """
        Create a graph representation of protein-ligand pair
        """
        # In a real implementation, this would create a detailed graph structure
        # with atomic coordinates, bonds, etc.
        
        # For this example, we'll create simplified data objects
        # Protein features (simplified)
        num_residues = len(protein_data) if isinstance(protein_data, str) else 100
        protein_nodes = torch.randn(num_residues, 16)  # Random features
        protein_edges = torch.randint(0, num_residues, (2, num_residues * 4))
        protein_pos = torch.randn(num_residues, 3)  # 3D positions
        
        # Ligand features (simplified)
        try:
            # Simple mock implementation - in real use would extract from SMILES
            mol = Chem.MolFromSmiles(ligand_data)
            num_atoms = mol.GetNumAtoms()
            ligand_nodes = torch.randn(num_atoms, 16)  # Random features
            ligand_edges = torch.randint(0, num_atoms, (2, num_atoms * 3))
            ligand_pos = torch.randn(num_atoms, 3)  # 3D positions
        except:
            # Fallback if SMILES is invalid
            num_atoms = 20
            ligand_nodes = torch.randn(num_atoms, 16)
            ligand_edges = torch.randint(0, num_atoms, (2, num_atoms * 3)) 
            ligand_pos = torch.randn(num_atoms, 3)
        
        # Create a single PyG Data object
        data = Data(
            protein_x=protein_nodes,
            protein_edge_index=protein_edges,
            protein_pos=protein_pos,
            ligand_x=ligand_nodes,
            ligand_edge_index=ligand_edges,
            ligand_pos=ligand_pos,
            molecule_idx=torch.tensor([idx]),
        )
        
        return data