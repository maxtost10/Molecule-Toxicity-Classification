def print_model_comparison(test_results):
    """Prints a clean comparison table of the results"""
    lgbm = test_results['lgbm']
    gat = test_results['gat']
    
    print("\n" + "="*45)
    print(f"{'METRIC':<15} | {'LGBM':<10} | {'GAT':<10}")
    print("-" * 45)
    print(f"{'F1-Score':<15} | {lgbm['f1']:.4f}     | {gat['f1']:.4f}")
    print(f"{'ROC-AUC':<15} | {lgbm['auc']:.4f}     | {gat['auc']:.4f}")
    print(f"{'Precision':<15} | {lgbm['precision']:.4f}     | {gat['precision']:.4f}")
    print(f"{'Recall':<15} | {lgbm['recall']:.4f}     | {gat['recall']:.4f}")
    print("="*45)
    
    winner = "GAT" if gat['f1'] > lgbm['f1'] else "LGBM"
    print(f"🏆 Best F1-Score: {winner}\n")