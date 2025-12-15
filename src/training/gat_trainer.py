import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from src.config import Config
from src.models.gat_net import HepatotoxicityGAT

def calculate_class_weights(targets):
    """Calculate class weights for imbalanced dataset"""
    unique, counts = np.unique(targets, return_counts=True)
    class_weights = len(targets) / (len(unique) * counts)
    
    print(f"Class distribution: {dict(zip(unique, counts))}")
    print(f"Class weights: {dict(zip(unique, class_weights))}")
    
    return torch.tensor(class_weights[1], dtype=torch.float)  # Weight for positive class

def train_gat_model(gnn_loaders, data_splits, node_features_dim, device='cuda'):
    """Train GAT model with weighted loss for class imbalance"""
    print(f"\n🏋️ TRAINING GAT MODEL")
    print("-" * 40)
    
    # Check device availability
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print(f"✓ Using CPU")
    
    # Calculate class weights for loss function
    train_targets = data_splits['train']['targets']
    pos_weight = calculate_class_weights(train_targets)
    
    # Initialize model
    model = HepatotoxicityGAT(
        node_features=node_features_dim,
        hidden_dim=Config.GAT_HIDDEN,
        num_heads=Config.GAT_HEADS,
        num_layers=Config.GAT_LAYERS,
        dropout=0.3
    ).to(device)
    
    print(f"\n🏗️ Model Architecture:")
    print(f"  - Node features: {node_features_dim}")
    print(f"  - Hidden dimension: {Config.GAT_HIDDEN}")
    print(f"  - Attention heads: {Config.GAT_HEADS}")
    print(f"  - Layers: {Config.GAT_LAYERS}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function with class weighting
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    # Training parameters
    num_epochs = Config.GAT_EPOCHS
    best_val_f1 = 0.0
    best_model_state = None
    patience_counter = 0
    patience = Config.GAT_PATIENCE
    
    train_losses = []
    val_f1_scores = []
    val_losses = []
    
    print(f"\n🚀 Starting training...")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Early stopping patience: {patience}")
    print(f"  - Class weight (pos): {pos_weight.item():.2f}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in gnn_loaders['train']:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out.squeeze(), batch.y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(gnn_loaders['train'])
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        val_probabilities = []
        
        with torch.no_grad():
            for batch in gnn_loaders['val']:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out.squeeze(), batch.y)
                val_loss += loss.item()
                
                # Collect predictions
                probs = torch.sigmoid(out.squeeze()).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                val_probabilities.extend(probs)
                val_predictions.extend(preds)
                val_targets.extend(batch.y.cpu().numpy())
        
        val_loss /= len(gnn_loaders['val'])
        val_losses.append(val_loss)
        
        # Calculate F1 score
        val_f1 = f1_score(val_targets, val_predictions)
        val_f1_scores.append(val_f1)
        
        # Learning rate scheduling
        scheduler.step(val_f1)
        
        # Early stopping and best model saving
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val F1: {val_f1:.4f} | "
                  f"Best F1: {best_val_f1:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {elapsed_time/60:.1f}m")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏰ Early stopping at epoch {epoch} (patience: {patience})")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    total_time = time.time() - start_time
    print(f"\n✅ Training completed!")
    print(f"  - Total time: {total_time/60:.1f} minutes")
    print(f"  - Best validation F1: {best_val_f1:.4f}")
    print(f"  - Final learning rate: {optimizer.param_groups[0]['lr']:.2e}")
    
    # Plot training curves
    os.makedirs('./Figures', exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    epochs_range = range(len(train_losses))
    
    # Loss curves
    axes[0].plot(epochs_range, train_losses, label='Train Loss', color='blue')
    axes[0].plot(epochs_range, val_losses, label='Val Loss', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # F1 score curve
    axes[1].plot(epochs_range, val_f1_scores, label='Val F1', color='green')
    axes[1].axhline(y=best_val_f1, color='green', linestyle='--', alpha=0.7, label=f'Best F1: {best_val_f1:.4f}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Validation F1 Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate curve
    # Approximation for visualization
    axes[2].plot(epochs_range, [0.001 * (0.5 ** (epoch // 10)) for epoch in epochs_range], color='purple')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./Figures/gat_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close() # Close plot to free memory
    
    # Final validation metrics
    model.eval()
    final_val_predictions = []
    final_val_probabilities = []
    final_val_targets = []
    
    with torch.no_grad():
        for batch in gnn_loaders['val']:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            
            probs = torch.sigmoid(out.squeeze()).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            final_val_probabilities.extend(probs)
            final_val_predictions.extend(preds)
            final_val_targets.extend(batch.y.cpu().numpy())
    
    # Calculate comprehensive metrics
    val_f1 = f1_score(final_val_targets, final_val_predictions)
    val_auc = roc_auc_score(final_val_targets, final_val_probabilities)
    
    # Precision and recall
    tn, fp, fn, tp = confusion_matrix(final_val_targets, final_val_predictions).ravel()
    val_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    val_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\n📊 GAT Validation Performance:")
    print(f"  - F1-score: {val_f1:.4f}")
    print(f"  - ROC-AUC: {val_auc:.4f}")
    print(f"  - Precision: {val_precision:.4f}")
    print(f"  - Recall: {val_recall:.4f}")
    print(f"  - Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    gat_results = {
        'model': model,
        'device': device,
        'val_f1': val_f1,
        'val_auc': val_auc,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_predictions': final_val_predictions,
        'val_probabilities': final_val_probabilities,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_f1_scores': val_f1_scores
        },
        'training_time': total_time
    }
    
    return gat_results