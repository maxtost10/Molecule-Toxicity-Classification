class Config:
    """
    Global configuration for Liver Toxicity Prediction Project.
    Toggles between QUICK_TEST mode for debugging and full training mode.
    """
    # Global test mode
    QUICK_TEST = True
    
    # Data settings
    MAX_MOLECULES = 3500 if QUICK_TEST else None
    
    # LGBM settings
    LGBM_ROUNDS = 50 if QUICK_TEST else 1000
    LGBM_EARLY_STOP = 10 if QUICK_TEST else 50
    LGBM_LEAVES = 15 if QUICK_TEST else 31
    
    # GAT settings  
    GAT_EPOCHS = 1 if QUICK_TEST else 6
    GAT_PATIENCE = 2 if QUICK_TEST else 3
    GAT_HIDDEN = 8 if QUICK_TEST else 128
    GAT_HEADS = 2 if QUICK_TEST else 5
    GAT_LAYERS = 2 if QUICK_TEST else 4
    
    # Feature settings
    MORGAN_BITS = 512 if QUICK_TEST else 2048
    BATCH_SIZE = 25 if QUICK_TEST else 64
    
    # Analysis settings
    SHAP_SAMPLES = 50 if QUICK_TEST else 100
    TOP_FEATURES = 10