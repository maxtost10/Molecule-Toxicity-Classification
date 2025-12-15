import os
import shap
import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from src.config import Config

def analyze_lgbm_interpretability(lgbm_results, lgbm_data, top_n=None):
    """Analyze LGBM feature importance using SHAP"""
    if top_n is None:
        top_n = Config.TOP_FEATURES
        
    print(f"\n🔍 LGBM INTERPRETABILITY ANALYSIS")
    print("-" * 40)
    
    if lgbm_results is None:
        return
    
    model = lgbm_results['model']
    feature_names = lgbm_results['feature_names']
    
    # Get test data for SHAP analysis
    X_test = lgbm_data['test']['features']
    
    # Use a subset for faster computation
    shap_sample_size = min(Config.SHAP_SAMPLES, X_test.shape[0])
    X_shap = X_test[:shap_sample_size]
    
    try:
        # TreeExplainer for LightGBM
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        mean_shap_importance = np.mean(np.abs(shap_values), axis=0)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': mean_shap_importance
        }).sort_values('shap_importance', ascending=False)
        
        # Save figure
        os.makedirs('./Figures', exist_ok=True)
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(top_n)
        plt.barh(range(len(top_features)), top_features['shap_importance'], color='skyblue')
        plt.yticks(range(len(top_features)), [f.replace('Morgan_bit_', 'M') for f in top_features['feature']])
        plt.xlabel('Mean |SHAP value|')
        plt.title(f'Top {top_n} Feature Importance (SHAP)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('./Figures/lgbm_interpretability.png', dpi=300)
        plt.close()
        
        return {'feature_importance': importance_df}

    except Exception as e:
        print(f"❌ SHAP analysis failed: {str(e)}")
        return None

def visualize_attention_weights(model, sample_molecules, device, num_examples=3):
    """Visualize GAT attention weights for sample molecules"""
    print(f"\n🔍 GAT ATTENTION VISUALIZATION")
    
    model.eval()
    os.makedirs('./Figures', exist_ok=True)
    
    # Try to find one pos and one neg example
    try:
        hepatotoxic_idx = next(i for i, mol in enumerate(sample_molecules) if mol.y.item() == 1)
        non_hepatotoxic_idx = next(i for i, mol in enumerate(sample_molecules) if mol.y.item() == 0)
        examples = [(hepatotoxic_idx, "Hepatotoxic"), (non_hepatotoxic_idx, "Non-hepatotoxic")]
    except StopIteration:
        # Fallback if we can't find both classes
        examples = [(0, "Sample 1"), (1, "Sample 2")]

    fig, axes = plt.subplots(len(examples), 2, figsize=(12, 5 * len(examples)))
    if len(examples) == 1: axes = [axes] # Handle single example case
    
    for row, (mol_idx, label) in enumerate(examples):
        mol = sample_molecules[mol_idx]
        batch_data = Data(x=mol.x, edge_index=mol.edge_index, batch=torch.zeros(mol.x.shape[0], dtype=torch.long)).to(device)
        
        with torch.no_grad():
            out, attention_weights = model(batch_data.x, batch_data.edge_index, batch_data.batch, return_attention_weights=True)
            prob = torch.sigmoid(out).item()
        
        G = to_networkx(mol, to_undirected=True, remove_self_loops=True)
        pos = nx.spring_layout(G, seed=42)
        
        # Color nodes by attention
        if attention_weights:
            edge_index_att, att_weights = attention_weights[0]
            node_importance = torch.zeros(mol.x.shape[0])
            att_weights_cpu = att_weights.cpu()
            edge_index_cpu = edge_index_att.cpu()
            
            for i in range(edge_index_cpu.shape[1]):
                node_importance[edge_index_cpu[0, i].item()] += att_weights_cpu[i].mean().item()
            
            if node_importance.max() > 0: node_importance /= node_importance.max()
            node_colors = plt.cm.Reds(node_importance.numpy())
        else:
            node_colors = 'lightblue'
            
        ax1 = axes[row][0] if len(examples) > 1 else axes[0]
        nx.draw(G, pos, ax=ax1, node_color=node_colors, node_size=300, with_labels=True)
        ax1.set_title(f"{label}\nPred: {prob:.3f}")
        
    plt.tight_layout()
    plt.savefig('./Figures/gat_attention_visualization.png', dpi=300)
    plt.close()

def create_comprehensive_comparison_plots(test_results, lgbm_results, gat_results, node_features_dim):
    """Create comprehensive comparison visualizations"""
    print(f"\n📊 CREATING COMPARISON VISUALIZATIONS")
    
    os.makedirs('./Figures', exist_ok=True)
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 4)
    
    # 1. ROC Curves
    ax1 = fig.add_subplot(gs[0, 0])
    lgbm_fpr, lgbm_tpr, _ = roc_curve(test_results['targets'], test_results['lgbm']['probabilities'])
    gat_fpr, gat_tpr, _ = roc_curve(test_results['targets'], test_results['gat']['probabilities'])
    ax1.plot(lgbm_fpr, lgbm_tpr, label=f'LGBM ({test_results["lgbm"]["auc"]:.3f})')
    ax1.plot(gat_fpr, gat_tpr, label=f'GAT ({test_results["gat"]["auc"]:.3f})')
    ax1.plot([0,1],[0,1], 'k--')
    ax1.legend()
    ax1.set_title("ROC Curves")
    
    # 2. Precision-Recall
    ax2 = fig.add_subplot(gs[0, 1])
    lgbm_p, lgbm_r, _ = precision_recall_curve(test_results['targets'], test_results['lgbm']['probabilities'])
    gat_p, gat_r, _ = precision_recall_curve(test_results['targets'], test_results['gat']['probabilities'])
    ax2.plot(lgbm_r, lgbm_p, label='LGBM')
    ax2.plot(gat_r, gat_p, label='GAT')
    ax2.set_title("Precision-Recall")
    ax2.legend()

    # 3. Bar Chart Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    metrics = ['F1', 'AUC']
    x = np.arange(len(metrics))
    width = 0.35
    ax3.bar(x - width/2, [test_results['lgbm']['f1'], test_results['lgbm']['auc']], width, label='LGBM')
    ax3.bar(x + width/2, [test_results['gat']['f1'], test_results['gat']['auc']], width, label='GAT')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.set_title("Metrics")

    plt.tight_layout()
    plt.savefig('./Figures/comprehensive_model_comparison.png', dpi=300)
    plt.close()