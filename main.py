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
    create_comparison_plots,
    print_model_comparison
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
    create_comparison_plots(test_results)
    statistical_significance_testing(test_results)
    print_model_comparison(test_results)

    print("\n🔍 Generating Attention Visualization...")
    
    # 1. Find the first Ground Truth Positive (Toxic)
    pos_sample = next(d for d in dm.val_dataset if d.y.item() == 1.0)
    
    # 2. Find the first Ground Truth Negative (Non-Toxic)
    neg_sample = next(d for d in dm.val_dataset if d.y.item() == 0.0)
    
    # 3. Visualize
    visualize_attention_weights(model, [pos_sample, neg_sample], model.device)
    
    print("\n✅ Project execution finished.")

if __name__ == "__main__":
    main()