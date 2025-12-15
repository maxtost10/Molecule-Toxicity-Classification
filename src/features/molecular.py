import numpy as np
from src.config import Config

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("❌ RDKit not available - install with: conda install -c rdkit rdkit")

def extract_molecular_features(molecules, descriptor_names=None):
    """Extract molecular fingerprints and descriptors for LGBM"""
    print(f"\n🧪 Extracting molecular features for {len(molecules)} molecules...")
    
    if not RDKIT_AVAILABLE:
        print("❌ RDKit not available - cannot extract molecular features")
        return None, None
    
    # Define basic descriptors (liver-relevant)
    if descriptor_names is None:
        descriptor_names = [
            'MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
            'NumRotatableBonds', 'NumAromaticRings', 'NumHeteroatoms', 
            'HeavyAtomCount', 'RingCount'
        ]
    
    features = []
    feature_names = []
    
    # Generate feature names
    morgan_names = [f'Morgan_bit_{i}' for i in range(Config.MORGAN_BITS)]
    feature_names = morgan_names + descriptor_names
    
    print(f"  - Morgan fingerprints: {Config.MORGAN_BITS} bits")
    print(f"  - Molecular descriptors: {len(descriptor_names)} features")
    print(f"  - Total features: {len(feature_names)}")
    
    failed_count = 0
    for i, mol_data in enumerate(molecules):
        if i % 500 == 0:
            print(f"    Processed {i}/{len(molecules)} molecules...")
        
        try:
            # Handle both PyG objects (which have .smiles) and raw objects
            smiles = mol_data.smiles
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                failed_count += 1
                features.append(np.zeros(len(feature_names)))
                continue
            
            # Extract Morgan fingerprints
            morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=Config.MORGAN_BITS
            )
            morgan_bits = list(morgan_fp)
            
            # Extract molecular descriptors
            descriptors = []
            for desc_name in descriptor_names:
                try:
                    desc_value = getattr(Descriptors, desc_name)(mol)
                    descriptors.append(desc_value)
                except:
                    descriptors.append(0.0)
            
            # Combine features
            mol_features = morgan_bits + descriptors
            features.append(mol_features)
            
        except Exception as e:
            failed_count += 1
            features.append(np.zeros(len(feature_names)))
    
    features = np.array(features)
    
    print(f"✓ Feature extraction complete!")
    print(f"  - Successful: {len(molecules) - failed_count}/{len(molecules)} molecules")
    print(f"  - Failed: {failed_count} molecules (filled with zeros)")
    print(f"  - Feature matrix shape: {features.shape}")
    
    return features, feature_names

def prepare_lgbm_data(data_splits):
    """Prepare feature matrices for LGBM training"""
    print(f"\n🔧 PREPARING LGBM FEATURES")
    print("-" * 40)
    
    # Extract features for each split
    lgbm_data = {}
    feature_names = None
    
    for split_name, split_data in data_splits.items():
        print(f"\nExtracting features for {split_name} set...")
        features, feature_names = extract_molecular_features(split_data['molecules'])
        
        if features is not None:
            lgbm_data[split_name] = {
                'features': features,
                'targets': split_data['targets'],
                'indices': split_data['indices']
            }
        else:
            print(f"❌ Failed to extract features for {split_name} set")
            return None, None
    
    return lgbm_data, feature_names