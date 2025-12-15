import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import numpy as np
from torch_geometric.data import Data

from src.config import Config
from src.data import LiverDataModule
from src.features import prepare_lgbm_data
from src.models import HepatotoxicityGAT
from src.training import train_lgbm_with_class_weights
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
    # 1. Setup
    pl.seed_everything(42)
    print("🚀 Starting Liver Toxicity Prediction Project")
    
    # Init DataModule
    dm = LiverDataModule()
    dm.prepare_data()
    dm.setup()

    # ---------------------------------------------------------
    # PHASE 1: LGBM BASELINE (Classic ML)
    # ---------------------------------------------------------
    print("\n🏗️ PHASE 1: LGBM BASELINE")
    # We access the raw splits inside the DM for the feature extractor
    lgbm_data, feature_names = prepare_lgbm_data(dm.splits)
    lgbm_results = train_lgbm_with_class_weights(lgbm_data, feature_names)
    
    if lgbm_results:
        analyze_lgbm_interpretability(lgbm_results, lgbm_data)

    # ---------------------------------------------------------
    # PHASE 2: GAT MODEL (PyTorch Lightning)
    # ---------------------------------------------------------
    print("\n🏗️ PHASE 2: GAT MODEL")

    # Calculate class weights dynamically
    train_targets = dm.splits['train']['targets']
    neg, pos = np.bincount(train_targets)
    pos_weight = float(neg) / pos
    print(f"   • Class imbalance weight: {pos_weight:.2f}")

    # Init Model
    model = HepatotoxicityGAT(
        node_features=dm.node_features_dim,
        pos_weight=pos_weight,
        hidden_dim=Config.GAT_HIDDEN,
        num_heads=Config.GAT_HEADS,
        num_layers=Config.GAT_LAYERS,
        dropout=0.3
    )

    # Init Trainer
    checkpoint_callback = ModelCheckpoint(monitor='val_f1', mode='max', save_top_k=1, verbose=True)
    early_stop_callback = EarlyStopping(monitor='val_f1', mode='max', patience=Config.GAT_PATIENCE)

    trainer = pl.Trainer(
        max_epochs=Config.GAT_EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='auto',  # Automatically detects GPU/CPU
        devices='auto',
        enable_progress_bar=True,
        log_every_n_steps=10
    )

    # Train
    trainer.fit(model, dm)

    # Load best model for analysis
    if checkpoint_callback.best_model_path:
        print(f"   • Loading best model from: {checkpoint_callback.best_model_path}")
        model = HepatotoxicityGAT.load_from_checkpoint(
            checkpoint_callback.best_model_path, 
            weights_only=False
        )
    
    # ---------------------------------------------------------
    # PHASE 3: ANALYSIS & COMPARISON
    # ---------------------------------------------------------
    print("\n📊 PHASE 3: FINAL ANALYSIS")
    
    # Construct results dictionary to bridge PL with your analysis module
    gat_results = {
        'model': model,
        'device': model.device, 
        'training_time': 0, # PL doesn't track this by default, but minor detail
        'val_f1': trainer.callback_metrics.get('val_f1', 0).item()
    }
    
    # Create loader dictionary for the analysis tools
    gnn_loaders = {'test': dm.test_dataloader(), 'val': dm.val_dataloader()}

    # Run evaluation pipeline
    test_results = evaluate_on_test_set(lgbm_results, gat_results, lgbm_data, gnn_loaders)
    create_comprehensive_comparison_plots(test_results, lgbm_results, gat_results, dm.node_features_dim)
    statistical_significance_testing(test_results)
    generate_final_insights(test_results, lgbm_results, gat_results)
    create_project_summary_report(test_results)

    # Optional: GAT Attention Visualization
    # We grab a few samples from the validation split manually
    print("\n🔍 Generating Attention Visualization...")
    val_molecules = dm.splits['val']['molecules'][:20]
    # Small helper to wrap raw objects into Data objects with labels for the vis function
    sample_molecules = []
    for i, mol in enumerate(val_molecules):
        mol_copy = Data(
            x=mol.x.clone().to(torch.float), edge_index=mol.edge_index.clone(),
            edge_attr=mol.edge_attr.clone() if mol.edge_attr is not None else None,
            y=torch.tensor([dm.splits['val']['targets'][i]]),
            smiles=mol.smiles if hasattr(mol, 'smiles') else None
        )
        sample_molecules.append(mol_copy)
    
    visualize_attention_weights(model, sample_molecules, model.device)
    
    print("\n✅ Project execution finished.")

if __name__ == "__main__":
    main()