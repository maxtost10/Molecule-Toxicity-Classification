import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem

def prepare_gnn_data(data_splits, batch_size=32):
    """Prepare PyTorch Geometric data loaders from existing splits"""
    print(f"\n🔧 PREPARING GNN DATA LOADERS")
    print("-" * 40)
    
    loaders = {}
    
    for split_name, split_data in data_splits.items():
        molecules = split_data['molecules']
        targets = split_data['targets']
        
        print(f"\nPreparing {split_name} loader...")
        print(f"  - Molecules: {len(molecules)}")
        print(f"  - Batch size: {batch_size}")
        
        # Convert to PyTorch tensors and add targets to data objects
        processed_molecules = []
        for i, mol in enumerate(molecules):
            # Create a copy of the molecule data
            mol_copy = Data(
                x=mol.x.clone().to(torch.float),
                edge_index=mol.edge_index.clone(),
                edge_attr=mol.edge_attr.clone() if mol.edge_attr is not None else None,
                y=torch.tensor([targets[i]], dtype=torch.float),
                smiles=mol.smiles if hasattr(mol, 'smiles') else None
            )
            processed_molecules.append(mol_copy)
        
        # Create data loader
        shuffle = (split_name == 'train')
        loader = DataLoader(processed_molecules, batch_size=batch_size, shuffle=shuffle)
        loaders[split_name] = loader
        
        print(f"  ✓ {split_name} loader: {len(loader)} batches")
    
    return loaders

# --- Helper functions for inference on raw SMILES ---

def atom_features(atom):
    # Example: one-hot for atom type (C, N, O, S, F, Cl, others), degree, aromaticity
    atom_type_list = ['C', 'N', 'O', 'S', 'F', 'Cl']
    atom_type = [int(atom.GetSymbol() == s) for s in atom_type_list]
    atom_type.append(int(atom.GetSymbol() not in atom_type_list))
    degree = [atom.GetDegree() / 4]  # Normalized degree (0-4)
    aromatic = [int(atom.GetIsAromatic())]
    return torch.tensor(atom_type + degree + aromatic, dtype=torch.float)

def bond_features(bond):
    # Example: one-hot for bond type (SINGLE, DOUBLE, TRIPLE, AROMATIC)
    bond_type = [0, 0, 0, 0]
    if bond is not None:
        b = bond.GetBondType()
        bond_type = [
            int(b == Chem.rdchem.BondType.SINGLE),
            int(b == Chem.rdchem.BondType.DOUBLE),
            int(b == Chem.rdchem.BondType.TRIPLE),
            int(b == Chem.rdchem.BondType.AROMATIC)
        ]
    return torch.tensor(bond_type, dtype=torch.float)

def smiles_to_pyg_data_rdkit(smiles):
    """Convert SMILES string to PyTorch Geometric Data object"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Nodes
    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()])
    # Edges
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # Add both directions
        edge_index += [[start, end], [end, start]]
        edge_attr += [bond_features(bond), bond_features(bond)]
    if len(edge_index) == 0:
        # Handle single-atom molecules (rare)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    return data