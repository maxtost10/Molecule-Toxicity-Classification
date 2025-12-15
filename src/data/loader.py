import numpy as np
import torch
from torch_geometric.datasets import MoleculeNet
from sklearn.model_selection import train_test_split
from src.config import Config

def load_and_prepare_liver_data():
    """Load Tox21 dataset and extract liver toxicity subset"""
    print("\n📥 PHASE 1: DATA PREPARATION")
    print("-" * 50)
    
    # Load full dataset
    print("Loading Tox21 dataset...")
    # Using ./data as the root for consistency with project structure
    dataset = MoleculeNet(root='./data', name='Tox21')
    
    # Extract all labels and reshape
    all_labels = []
    for i in range(len(dataset)):
        labels = dataset[i].y.numpy().reshape(-1)
        all_labels.append(labels)
    all_labels = np.array(all_labels)
    
    # Focus on NR-AhR task (liver toxicity, index 2)
    liver_task_idx = 2
    liver_labels = all_labels[:, liver_task_idx]
    valid_mask = ~np.isnan(liver_labels)
    
    print(f"✓ Full dataset: {len(dataset):,} molecules")
    print(f"✓ Valid liver labels: {np.sum(valid_mask):,} molecules ({np.sum(valid_mask)/len(dataset)*100:.1f}%)")
    
    # Extract valid samples
    valid_indices = np.where(valid_mask)[0]
    
    # QUICK TEST MODE: Sample subset
    if Config.MAX_MOLECULES and len(valid_indices) > Config.MAX_MOLECULES:
        print(f"🚀 QUICK TEST MODE: Sampling {Config.MAX_MOLECULES} molecules from {len(valid_indices)}")
        np.random.seed(42)
        valid_indices = np.random.choice(valid_indices, Config.MAX_MOLECULES, replace=False)
    
    liver_molecules = [dataset[i] for i in valid_indices]
    liver_targets = liver_labels[valid_indices].astype(int)
    
    # Class distribution
    n_hepatotoxic = np.sum(liver_targets == 1)
    n_non_hepatotoxic = np.sum(liver_targets == 0)
    
    print(f"✓ Final dataset size: {len(liver_targets):,} molecules")
    print(f"✓ Class distribution:")
    print(f"  - Non-hepatotoxic: {n_non_hepatotoxic:,} ({n_non_hepatotoxic/len(liver_targets)*100:.1f}%)")
    print(f"  - Hepatotoxic: {n_hepatotoxic:,} ({n_hepatotoxic/len(liver_targets)*100:.1f}%)")
    print(f"  - Imbalance ratio: {n_non_hepatotoxic/n_hepatotoxic:.1f}:1")
    
    return liver_molecules, liver_targets, valid_indices

def stratified_split_data(molecules, targets):
    """Create stratified train/val/test splits"""
    print(f"\n🔀 Creating stratified splits...")
    
    # First split: train+val (85%) vs test (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        range(len(molecules)), targets, 
        test_size=0.15, random_state=42, stratify=targets
    )
    
    # Second split: train (70% of total) vs val (15% of total)
    # 0.15 / 0.85 ≈ 0.1764
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.176, random_state=42, stratify=y_temp
    )
    
    # Extract molecule objects
    train_molecules = [molecules[i] for i in X_train]
    val_molecules = [molecules[i] for i in X_val]
    test_molecules = [molecules[i] for i in X_test]
    
    print(f"✓ Train set: {len(train_molecules):,} molecules ({np.sum(y_train==1)} hepatotoxic, {np.sum(y_train==1)/len(y_train)*100:.1f}%)")
    print(f"✓ Val set: {len(val_molecules):,} molecules ({np.sum(y_val==1)} hepatotoxic, {np.sum(y_val==1)/len(y_val)*100:.1f}%)")
    print(f"✓ Test set: {len(test_molecules):,} molecules ({np.sum(y_test==1)} hepatotoxic, {np.sum(y_test==1)/len(y_test)*100:.1f}%)")
    
    splits = {
        'train': {'molecules': train_molecules, 'targets': y_train, 'indices': X_train},
        'val': {'molecules': val_molecules, 'targets': y_val, 'indices': X_val},
        'test': {'molecules': test_molecules, 'targets': y_test, 'indices': X_test}
    }
    
    return splits