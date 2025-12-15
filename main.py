import torch
import numpy as np
import matplotlib.pyplot as plt
from src.config import Config

# Import modules from our new package structure
from src.data import load_and_prepare_liver_data, stratified_split_data
from src.features import prepare_lgbm_data, prepare_gnn_data
from src.training import train_lgbm_with_class_weights, train_gat_model
from src.analysis import (
    evaluate_on_test_set, 
    statistical_significance_testing,
    analyze_lgbm_interpretability, 
    visualize_attention_weights, 
    create_comprehensive_comparison_plots,
    generate_final_insights,
    create_project_summary_report
)

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("🚀 Starting Liver Toxicity Prediction Project - MODULARIZED VERSION")
    print("📊 Target: NR-AhR hepatotoxicity (GAT vs LGBM comparison)")
    print(f"⚡ Mode: {'QUICK TEST' if Config.QUICK_TEST else 'FULL TRAINING'}")
    print("="*80)

    # ---------------------------------------------------------
    # PHASE 1: DATA PREPARATION
    # ---------------------------------------------------------
    molecules, targets, valid_indices = load_and_prepare_liver_data()
    data_splits = stratified_split_data(molecules, targets)

    # ---------------------------------------------------------
    # PHASE 2: LGBM BASELINE
    # ---------------------------------------------------------
    # Prepare features
    lgbm_data, feature_names = prepare_lgbm_data(data_splits)
    
    if lgbm_data is None:
        print("❌ LGBM data preparation failed. Exiting.")
        return

    # Train model
    lgbm_results = train_lgbm_with_class_weights(lgbm_data, feature_names)

    # Analyze interpretability (SHAP)
    if lgbm_results:
        analyze_lgbm_interpretability(lgbm_results, lgbm_data)
        print(f"✅ LGBM BASELINE COMPLETE!")
        print(f"📊 Best validation F1-score: {lgbm_results['val_f1']:.4f}")

    # ---------------------------------------------------------
    # PHASE 3: GAT MODEL
    # ---------------------------------------------------------
    # Prepare graph data
    gnn_loaders = prepare_gnn_data(data_splits, batch_size=Config.BATCH_SIZE)
    
    # Get node feature dimension from a sample
    sample_mol = data_splits['train']['molecules'][0]
    node_features_dim = sample_mol.x.shape[1]
    
    # Train model
    gat_results = train_gat_model(gnn_loaders, data_splits, node_features_dim)
    
    # Visualize attention on validation set samples
    if gat_results:
        # Prepare a small subset of molecules for visualization
        val_molecules = data_splits['val']['molecules']
        # We need to wrap them in Data objects if they aren't already fully compatible for the visualizer
        # (The visualizer expects a list of PyG Data objects with .y attributes)
        from torch_geometric.data import Data
        sample_molecules = []
        for i, mol in enumerate(val_molecules[:20]):
            mol_with_target = Data(
                x=mol.x.to(torch.float),
                edge_index=mol.edge_index,
                edge_attr=mol.edge_attr,
                y=torch.tensor([data_splits['val']['targets'][i]], dtype=torch.float),
                smiles=mol.smiles if hasattr(mol, 'smiles') else None
            )
            sample_molecules.append(mol_with_target)

        visualize_attention_weights(gat_results['model'], sample_molecules, gat_results['device'])
        
        print(f"\n✅ GAT MODEL COMPLETE!")
        print(f"📊 Best validation F1-score: {gat_results['val_f1']:.4f}")

    # ---------------------------------------------------------
    # PHASE 4: COMPARISON & ANALYSIS
    # ---------------------------------------------------------
    if lgbm_results and gat_results:
        # Evaluate on held-out test set
        test_results = evaluate_on_test_set(lgbm_results, gat_results, lgbm_data, gnn_loaders)
        
        # Create comparison plots
        create_comprehensive_comparison_plots(test_results, lgbm_results, gat_results, node_features_dim)
        
        # Statistical testing
        statistical_significance_testing(test_results)
        
        # Final reports
        generate_final_insights(test_results, lgbm_results, gat_results)
        create_project_summary_report(test_results)

    print("\n✅ Project execution finished.")

if __name__ == "__main__":
    main()