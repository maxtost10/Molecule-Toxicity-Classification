import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from src.config import Config

def simple_smote_alternative(X, y, random_state=42):
    """Simple SMOTE alternative using random oversampling with noise"""
    np.random.seed(random_state)
    
    # Separate classes
    minority_indices = np.where(y == 1)[0]
    majority_indices = np.where(y == 0)[0]
    
    minority_samples = X[minority_indices]
    
    # Calculate how many samples to generate
    n_majority = len(majority_indices)
    n_minority = len(minority_indices)
    n_to_generate = n_majority - n_minority
    
    if n_to_generate <= 0:
        return X, y
    
    print(f"    Generating {n_to_generate} synthetic minority samples...")
    
    # Generate synthetic samples by adding small noise to existing minority samples
    synthetic_samples = []
    for _ in range(n_to_generate):
        # Pick a random minority sample
        base_idx = np.random.choice(len(minority_samples))
        base_sample = minority_samples[base_idx].copy()
        
        # Add small Gaussian noise (only to non-binary features)
        # Assume first Morgan bits are binary, descriptors are continuous
        morgan_bits = Config.MORGAN_BITS
        
        # Keep Morgan fingerprint bits as is (binary)
        # Add noise only to molecular descriptors
        if len(base_sample) > morgan_bits:
            descriptor_part = base_sample[morgan_bits:]
            noise = np.random.normal(0, 0.1 * np.std(descriptor_part), len(descriptor_part))
            base_sample[morgan_bits:] += noise
            
            # Ensure non-negative values for descriptors that should be positive
            base_sample[morgan_bits:] = np.maximum(base_sample[morgan_bits:], 0)
        
        synthetic_samples.append(base_sample)
    
    # Combine original and synthetic data
    X_resampled = np.vstack([X, np.array(synthetic_samples)])
    y_resampled = np.hstack([y, np.ones(n_to_generate)])
    
    return X_resampled, y_resampled

def train_lgbm_with_class_weights(lgbm_data, feature_names):
    """Train LGBM model with class weights instead of SMOTE"""
    print(f"\n🏗️ PHASE 2: LGBM BASELINE WITH CLASS WEIGHTING")
    print("-" * 50)
    
    if lgbm_data is None:
        print("❌ No LGBM data available")
        return None
    
    # Get training data
    X_train = lgbm_data['train']['features']
    y_train = lgbm_data['train']['targets']
    X_val = lgbm_data['val']['features']
    y_val = lgbm_data['val']['targets']
    
    print(f"Original training data: {X_train.shape}")
    print(f"  - Class 0 (non-hepatotoxic): {np.sum(y_train==0)}")
    print(f"  - Class 1 (hepatotoxic): {np.sum(y_train==1)}")
    
    # Option 1: Use simple oversampling alternative
    print(f"\nApplying simple oversampling...")
    X_train_balanced, y_train_balanced = simple_smote_alternative(X_train, y_train)
    
    print(f"After oversampling: {X_train_balanced.shape}")
    print(f"  - Class 0 (non-hepatotoxic): {np.sum(y_train_balanced==0)}")
    print(f"  - Class 1 (hepatotoxic): {np.sum(y_train_balanced==1)}")
    
    # Calculate class weights for additional emphasis
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    scale_pos_weight = class_weights[1] / class_weights[0]
    
    print(f"  - Calculated scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Prepare LightGBM datasets
    train_data = lgb.Dataset(X_train_balanced, label=y_train_balanced)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': Config.LGBM_LEAVES,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 10,
        'scale_pos_weight': scale_pos_weight,  # Additional class weighting
        'random_state': 42,
        'verbose': -1
    }
    
    print(f"\nTraining LightGBM model...")
    print(f"Parameters: {params}")
    
    # Train model with early stopping
    lgbm_model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        num_boost_round=Config.LGBM_ROUNDS,
        callbacks=[
            lgb.early_stopping(stopping_rounds=Config.LGBM_EARLY_STOP),
            lgb.log_evaluation(period=max(1, Config.LGBM_ROUNDS // 10))
        ]
    )
    
    print(f"✓ Training completed!")
    print(f"  - Best iteration: {lgbm_model.best_iteration}")
    
    # Evaluate on validation set
    y_val_pred_proba = lgbm_model.predict(X_val, num_iteration=lgbm_model.best_iteration)
    y_val_pred = (y_val_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    val_f1 = f1_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    val_precision = np.sum((y_val_pred == 1) & (y_val == 1)) / np.sum(y_val_pred == 1) if np.sum(y_val_pred == 1) > 0 else 0
    val_recall = np.sum((y_val_pred == 1) & (y_val == 1)) / np.sum(y_val == 1)
    
    print(f"\n📊 LGBM Validation Performance:")
    print(f"  - F1-score: {val_f1:.4f}")
    print(f"  - ROC-AUC: {val_auc:.4f}")
    print(f"  - Precision: {val_precision:.4f}")
    print(f"  - Recall: {val_recall:.4f}")
    
    lgbm_results = {
        'model': lgbm_model,
        'val_f1': val_f1,
        'val_auc': val_auc,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_predictions': y_val_pred,
        'val_probabilities': y_val_pred_proba,
        'feature_names': feature_names,
        'oversampling_applied': True
    }
    
    return lgbm_results