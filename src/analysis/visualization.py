import os
import shap
import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from src.config import Config

def analyze_lgbm_interpretability(lgbm_results, lgbm_data, top_n=10):
    """Plot Top-N SHAP values for LGBM"""
    if not lgbm_results: return
    
    print(f"\n🔍 Generating LGBM SHAP Plot...")
    model = lgbm_results['model']
    X_test = lgbm_data['test']['features']
    
    # Use subset for speed
    X_shap = X_test[:min(Config.SHAP_SAMPLES, len(X_test))]
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
        if isinstance(shap_values, list): shap_values = shap_values[1]
            
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        feature_names = lgbm_results['feature_names']
        
        # Sort and plot
        indices = np.argsort(mean_shap)[-top_n:]
        plt.figure(figsize=(10, 6))
        plt.barh(range(top_n), mean_shap[indices], align='center', color='#4e79a7')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Mean |SHAP value|')
        plt.title('LGBM Feature Importance')
        plt.tight_layout()
        
        os.makedirs('./Figures', exist_ok=True)
        plt.savefig('./Figures/lgbm_shap.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"⚠️ SHAP failed: {e}")

def visualize_attention_weights(model, sample_molecules, device):
    """Visualizes attention for a few samples"""
    print(f"\n🔍 Generating GAT Attention Plot...")
    model.eval()
    os.makedirs('./Figures', exist_ok=True)
    
    # Plot first 2 samples
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    for i, mol in enumerate(sample_molecules[:2]):
        batch = Data(x=mol.x, edge_index=mol.edge_index, batch=torch.zeros(mol.x.shape[0], dtype=torch.long)).to(device)
        with torch.no_grad():
            out, att = model(batch.x, batch.edge_index, batch.batch, return_attention_weights=True)
            prob = torch.sigmoid(out).item()
            
        # Graph viz
        G = to_networkx(mol, to_undirected=True, remove_self_loops=True)
        pos = nx.spring_layout(G, seed=42)
        
        # Color by attention
        node_colors = 'lightblue'
        if att:
            edge_index, weights = att[0]
            importance = torch.zeros(mol.x.shape[0])
            for idx, w in enumerate(weights.cpu()):
                src = edge_index[0, idx]
                importance[src] += w.mean().item()
            node_colors = plt.cm.Reds(importance.numpy() / importance.max().item())

        nx.draw(G, pos, ax=axes[i], node_color=node_colors, with_labels=True, node_size=300)
        axes[i].set_title(f"Label: {int(mol.y.item())} | Pred: {prob:.2f}")

    plt.tight_layout()
    plt.savefig('./Figures/gat_attention.png', dpi=300)
    plt.close()

def create_comparison_plots(test_results):
    """Minimalist comparison: ROC and PR Curves only"""
    print(f"\n📊 Generating Comparison Curves...")
    os.makedirs('./Figures', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    models = ['lgbm', 'gat']
    colors = {'lgbm': 'blue', 'gat': 'red'}
    names = {'lgbm': 'LGBM', 'gat': 'GAT'}
    
    for m in models:
        y_true = test_results['targets']
        y_score = test_results[m]['probabilities']
        
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax1.plot(fpr, tpr, label=f"{names[m]} (AUC={auc(fpr, tpr):.3f})", color=colors[m], lw=2)
        
        # PR
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ax2.plot(rec, prec, label=f"{names[m]} (AP={np.mean(prec):.3f})", color=colors[m], lw=2)

    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('./Figures/model_comparison.png', dpi=300)
    plt.close()