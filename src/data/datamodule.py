import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch_geometric.transforms as T
from src.config import Config
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
        
        # Initialize the Virtual Node transform
        # This adds edges from a new node index to all existing nodes
        virtual_node_transform = T.VirtualNode()
        
        for i, mol in enumerate(molecules):
            # 1. Basic Conversion
            data = Data(
                x=mol.x.clone().to(torch.float),
                edge_index=mol.edge_index.clone(),
                edge_attr=mol.edge_attr.clone() if mol.edge_attr is not None else None,
                y=torch.tensor([targets[i]], dtype=torch.float),
                smiles=mol.smiles if hasattr(mol, 'smiles') else None
            )
            
            # 2. Add Virtual Node Topology (Edges)
            # This updates edge_index to include connections to the new node
            data = virtual_node_transform(data)
            
            processed_data.append(data)
            
        return processed_data

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=7)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=7)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=7)