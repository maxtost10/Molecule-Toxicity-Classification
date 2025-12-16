import numpy as np
import scipy.stats as stats
import torch
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix

def evaluate_on_test_set(lgbm_results, gat_results, lgbm_data, gnn_loaders):
    """Evaluate both models on the held-out test set"""
    print(f"\n🎯 FINAL TEST SET EVALUATION")
    print("-" * 40)
    
    # ===== LGBM TEST EVALUATION =====
    print(f"\n📈 Evaluating LGBM on test set...")
    
    X_test_lgbm = lgbm_data['test']['features']
    y_test = lgbm_data['test']['targets']
    
    # LGBM predictions
    lgbm_model = lgbm_results['model']
    y_test_pred_proba_lgbm = lgbm_model.predict(X_test_lgbm, num_iteration=lgbm_model.best_iteration)
    y_test_pred_lgbm = (y_test_pred_proba_lgbm > 0.5).astype(int)
    
    # LGBM metrics
    lgbm_test_f1 = f1_score(y_test, y_test_pred_lgbm)
    lgbm_test_auc = roc_auc_score(y_test, y_test_pred_proba_lgbm)
    lgbm_test_ap = average_precision_score(y_test, y_test_pred_proba_lgbm)
    
    # Confusion matrix for LGBM
    lgbm_cm = confusion_matrix(y_test, y_test_pred_lgbm)
    lgbm_tn, lgbm_fp, lgbm_fn, lgbm_tp = lgbm_cm.ravel()
    lgbm_precision = lgbm_tp / (lgbm_tp + lgbm_fp) if (lgbm_tp + lgbm_fp) > 0 else 0
    lgbm_recall = lgbm_tp / (lgbm_tp + lgbm_fn) if (lgbm_tp + lgbm_fn) > 0 else 0
    lgbm_specificity = lgbm_tn / (lgbm_tn + lgbm_fp) if (lgbm_tn + lgbm_fp) > 0 else 0
    
    print(f"✓ LGBM Test Results:")
    print(f"  - F1-score: {lgbm_test_f1:.4f}")
    print(f"  - ROC-AUC: {lgbm_test_auc:.4f}")
    
    # ===== GAT TEST EVALUATION =====
    print(f"\n🧠 Evaluating GAT on test set...")
    
    gat_model = gat_results['model']
    device = gat_results['device']
    gat_model.eval()
    
    gat_test_predictions = []
    gat_test_probabilities = []
    gat_test_targets = []
    
    with torch.no_grad():
        for batch in gnn_loaders['test']:
            batch = batch.to(device)
            out = gat_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.morgan_fp)
            
            probs = torch.sigmoid(out.squeeze()).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            gat_test_probabilities.extend(probs)
            gat_test_predictions.extend(preds)
            gat_test_targets.extend(batch.y.cpu().numpy())
    
    # GAT metrics
    gat_test_f1 = f1_score(gat_test_targets, gat_test_predictions)
    gat_test_auc = roc_auc_score(gat_test_targets, gat_test_probabilities)
    gat_test_ap = average_precision_score(gat_test_targets, gat_test_probabilities)
    
    # Confusion matrix for GAT
    gat_cm = confusion_matrix(gat_test_targets, gat_test_predictions)
    gat_tn, gat_fp, gat_fn, gat_tp = gat_cm.ravel()
    gat_precision = gat_tp / (gat_tp + gat_fp) if (gat_tp + gat_fp) > 0 else 0
    gat_recall = gat_tp / (gat_tp + gat_fn) if (gat_tp + gat_fn) > 0 else 0
    gat_specificity = gat_tn / (gat_tn + gat_fp) if (gat_tn + gat_fp) > 0 else 0
    
    print(f"✓ GAT Test Results:")
    print(f"  - F1-score: {gat_test_f1:.4f}")
    print(f"  - ROC-AUC: {gat_test_auc:.4f}")
    
    test_results = {
        'lgbm': {
            'predictions': y_test_pred_lgbm,
            'probabilities': y_test_pred_proba_lgbm,
            'f1': lgbm_test_f1,
            'auc': lgbm_test_auc,
            'ap': lgbm_test_ap,
            'precision': lgbm_precision,
            'recall': lgbm_recall,
            'specificity': lgbm_specificity,
            'confusion_matrix': lgbm_cm,
            'targets': y_test,
        },
        'gat': {
            'predictions': gat_test_predictions,
            'probabilities': gat_test_probabilities,
            'f1': gat_test_f1,
            'auc': gat_test_auc,
            'ap': gat_test_ap,
            'precision': gat_precision,
            'recall': gat_recall,
            'specificity': gat_specificity,
            'confusion_matrix': gat_cm,
            'targets': gat_test_targets,
        }
    }
    
    return test_results