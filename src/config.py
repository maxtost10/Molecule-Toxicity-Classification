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
    GAT_EPOCHS = 20 if QUICK_TEST else 100
    GAT_PATIENCE = 10 if QUICK_TEST else 20
    GAT_HIDDEN = 32 if QUICK_TEST else 128
    GAT_HEADS = 2 if QUICK_TEST else 5
    GAT_LAYERS = 2 if QUICK_TEST else 4
    
    # Feature settings
    MORGAN_BITS = 512 if QUICK_TEST else 2048
    BATCH_SIZE = 25 if QUICK_TEST else 64
    
    # Analysis settings
    SHAP_SAMPLES = 50 if QUICK_TEST else 100
    TOP_FEATURES = 10