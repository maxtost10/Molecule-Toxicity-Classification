import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch_geometric.transforms as T
from rdkit.Chem import rdFingerprintGenerator
from src.config import Config
from rdkit import Chem
from src.data.loader import load_and_prepare_liver_data, stratified_split_data

class LiverDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = Config.BATCH_SIZE
        self.molecules = None
        self.targets = None
        self.splits = None
        
        # Placeholders for datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        print("⚡ [DataModule] Preparing data...")
        load_and_prepare_liver_data()

    def setup(self, stage=None):
        print("⚡ [DataModule] Setting up datasets with Virtual Nodes...")
        
        # 1. Load raw data
        self.molecules, self.targets, _ = load_and_prepare_liver_data()
        
        # 2. Split data
        self.splits = stratified_split_data(self.molecules, self.targets)
        
        # 3. Convert to PyG Data objects (with Virtual Nodes!)
        self.train_dataset = self._convert_to_pyg(self.splits['train'])
        self.val_dataset = self._convert_to_pyg(self.splits['val'])
        self.test_dataset = self._convert_to_pyg(self.splits['test'])
        
        # Calculate node feature dimension for model init
        self.node_features_dim = self.train_dataset[0].x.shape[1]

    def _convert_to_pyg(self, split_data):
        """Helper to convert raw molecule objects to PyG Data list with Virtual Nodes"""
        processed_data = []
        molecules = split_data['molecules']
        targets = split_data['targets']
        
        virtual_node_transform = T.VirtualNode()
        
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, 
            fpSize=Config.MORGAN_BITS
        )
        
        for i, mol in enumerate(molecules):
            # 1. Check for valid node features (mol.x)
            # Ensure x exists and has nodes
            if mol.x is None or mol.x.shape[0] == 1:
                continue

            # 2. Generate Morgan Fingerprint
            # We need the RDKit object. If 'mol' is just a container, we rebuild from SMILES.
            rdkit_mol = Chem.MolFromSmiles(mol.smiles)
            if rdkit_mol is None:
                continue # Skip invalid smiles
            
            # 3. Use the Generator
            fp = morgan_gen.GetFingerprint(rdkit_mol)
            
            # Convert to Tensor
            morgan_tensor = torch.tensor(list(fp), dtype=torch.float).unsqueeze(0) # Shape: [1, nBits]

            # 4. Create PyG Data Object
            data = Data(
                x=mol.x.clone().to(torch.long), 
                edge_index=mol.edge_index.clone(),
                edge_attr=mol.edge_attr.clone() if mol.edge_attr is not None else None,
                y=torch.tensor([targets[i]], dtype=torch.float),
                morgan_fp=morgan_tensor,
                smiles=mol.smiles
            )
            
            # 5. Add Virtual Node (Topology + Feature Row)
            data = virtual_node_transform(data)
            
            processed_data.append(data)
            
        return processed_data

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=7)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=7)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=7)