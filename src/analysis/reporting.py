def generate_final_insights(test_results, lgbm_results, gat_results):
    """Generate final insights and recommendations"""
    print(f"\n🎯 FINAL INSIGHTS & RECOMMENDATIONS")
    print("=" * 60)
    
    lgbm_score = test_results['lgbm']['f1']
    gat_score = test_results['gat']['f1']
    
    print(f"\n🏆 OVERALL PERFORMANCE:")
    if gat_score > lgbm_score:
        print(f"✓ Winner: GAT (+{gat_score - lgbm_score:.4f})")
    else:
        print(f"✓ Winner: LGBM (+{lgbm_score - gat_score:.4f})")
        
    print(f"\n⚡ Efficiency:")
    print(f"   • LGBM: Fast training, simpler model")
    print(f"   • GAT: {gat_results['training_time']/60:.1f} min training")

def create_project_summary_report(test_results):
    print(f"\n🎯 FINAL TEST RESULTS SUMMARY:")
    print(f"   Model    │ F1-Score │ ROC-AUC")
    print(f"   ─────────┼──────────┼────────")
    print(f"   LGBM     │  {test_results['lgbm']['f1']:.4f}  │ {test_results['lgbm']['auc']:.4f}")
    print(f"   GAT      │  {test_results['gat']['f1']:.4f}  │ {test_results['gat']['auc']:.4f}")