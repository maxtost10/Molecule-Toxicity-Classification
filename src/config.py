class Config:
    """
    Global configuration for Liver Toxicity Prediction Project.
    Toggles between QUICK_TEST mode for debugging and full training mode.
    """
    # Global test mode
    QUICK_TEST = True  # Set to False for full training
    
    # Data settings
    MAX_MOLECULES = 200 if QUICK_TEST else None
    
    # LGBM settings
    LGBM_ROUNDS = 50 if QUICK_TEST else 1000
    LGBM_EARLY_STOP = 10 if QUICK_TEST else 50
    LGBM_LEAVES = 15 if QUICK_TEST else 31
    
    # GAT settings  
    GAT_EPOCHS = 2 if QUICK_TEST else 100
    GAT_PATIENCE = 5 if QUICK_TEST else 20
    GAT_HIDDEN = 2 if QUICK_TEST else 64
    GAT_HEADS = 1 if QUICK_TEST else 4
    GAT_LAYERS = 2 if QUICK_TEST else 3
    
    # Feature settings
    MORGAN_BITS = 512 if QUICK_TEST else 2048
    BATCH_SIZE = 32
    
    # Analysis settings
    SHAP_SAMPLES = 50 if QUICK_TEST else 100
    TOP_FEATURES = 10