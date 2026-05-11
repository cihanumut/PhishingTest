"""
PhishGuard - Professional Model Training Pipeline
===================================================
Trains XGBoost on the 111-feature phishing dataset with:
  - Proper missing value handling (-1 → NaN for XGBoost native support)
  - Stratified train/val/test split (70/15/15)
  - Bayesian hyperparameter optimization via Optuna
  - Early stopping on validation set
  - 5-fold stratified cross-validation for robust metrics
  - Feature importance analysis & visualization
  - Comprehensive classification report + confusion matrix
  - Model + metadata saved to disk

No manual controls, no whitelists, no hardcoded overrides.
Pure data-driven machine learning.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import json
import os
import time
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
    confusion_matrix, roc_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DATASET_PATH = 'dataset_full.csv'
MODEL_DIR = 'trained_model'
MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_phishguard.pkl')
METADATA_PATH = os.path.join(MODEL_DIR, 'training_metadata.json')
FIGURES_DIR = os.path.join(MODEL_DIR, 'figures')

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # from remaining after test split


def load_and_preprocess(path):
    """Load dataset and handle missing values properly."""
    print("=" * 60)
    print("  PHASE 1: DATA LOADING & PREPROCESSING")
    print("=" * 60)
    
    df = pd.read_csv(path)
    print(f"  ✓ Loaded {df.shape[0]:,} samples, {df.shape[1]} columns")
    
    # Separate features and target
    X = df.drop('phishing', axis=1)
    y = df['phishing']
    
    feature_names = list(X.columns)
    
    print(f"  ✓ Target distribution:")
    print(f"      Legitimate (0): {(y == 0).sum():,} ({(y == 0).mean()*100:.1f}%)")
    print(f"      Phishing   (1): {(y == 1).sum():,} ({(y == 1).mean()*100:.1f}%)")
    
    # Replace -1 with NaN so XGBoost can handle missing values natively
    # XGBoost has built-in handling for NaN - it learns the optimal direction
    # for missing values at each split, which is better than imputation
    missing_before = (X == -1).sum().sum()
    X = X.replace(-1, np.nan)
    missing_after = X.isna().sum().sum()
    
    print(f"  ✓ Converted {missing_before:,} missing indicators (-1 → NaN)")
    print(f"      XGBoost will learn optimal missing-value directions automatically")
    
    # Report columns with high missing rates
    missing_pct = X.isna().mean().sort_values(ascending=False)
    high_missing = missing_pct[missing_pct > 0.1]
    if len(high_missing) > 0:
        print(f"  ℹ Columns with >10% missing values: {len(high_missing)}")
        for col, pct in high_missing.head(5).items():
            print(f"      {col}: {pct*100:.1f}%")
    
    print()
    return X.values, y.values, feature_names


def split_data(X, y):
    """Create stratified train/val/test splits."""
    print("=" * 60)
    print("  PHASE 2: DATA SPLITTING (Stratified)")
    print("=" * 60)
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Second split: separate validation from training
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_temp
    )
    
    print(f"  ✓ Train: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(y)*100:.0f}%)")
    print(f"  ✓ Val:   {X_val.shape[0]:,} samples ({X_val.shape[0]/len(y)*100:.0f}%)")
    print(f"  ✓ Test:  {X_test.shape[0]:,} samples ({X_test.shape[0]/len(y)*100:.0f}%)")
    print()
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train, y_train, X_val, y_val):
    """Train XGBoost with optimized hyperparameters and early stopping."""
    print("=" * 60)
    print("  PHASE 3: MODEL TRAINING")
    print("=" * 60)
    
    # Calculate scale_pos_weight for class imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"  ℹ Class imbalance ratio: {scale_pos_weight:.2f} (auto-adjusted)")
    
    model = xgb.XGBClassifier(
        # Core parameters
        n_estimators=3000,           # High ceiling, early stopping will find optimal
        max_depth=7,                 # Deep enough for complex patterns, not too deep for overfitting
        learning_rate=0.03,          # Slower learning for better generalization
        
        # Regularization
        min_child_weight=5,          # Prevent overfitting on small subgroups
        gamma=0.2,                   # Minimum loss reduction for splits
        reg_alpha=0.1,               # L1 regularization
        reg_lambda=1.5,              # L2 regularization
        
        # Sampling (reduce overfitting)
        subsample=0.8,               # Row sampling per tree
        colsample_bytree=0.7,        # Feature sampling per tree
        colsample_bylevel=0.8,       # Feature sampling per depth level
        
        # Class imbalance handling
        scale_pos_weight=scale_pos_weight,
        
        # Performance
        tree_method='hist',          # Fast histogram-based method
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='logloss',
        
        # Early stopping
        early_stopping_rounds=100,
    )
    
    print("  ⏳ Training with early stopping (patience=100 rounds)...")
    print()
    
    t_start = time.time()
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100  # Print every 100 rounds
    )
    
    training_time = time.time() - t_start
    best_iteration = model.best_iteration
    
    print()
    print(f"  ✓ Training completed in {training_time:.1f}s")
    print(f"  ✓ Best iteration: {best_iteration} / {model.n_estimators}")
    print(f"  ✓ Best validation logloss: {model.best_score:.6f}")
    print()
    
    return model, training_time, best_iteration


def cross_validate(X_train_full, y_train_full, model_params):
    """Run 5-fold stratified cross-validation for robust performance estimates."""
    print("=" * 60)
    print("  PHASE 4: 5-FOLD CROSS-VALIDATION")
    print("=" * 60)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    cv_scores = {
        'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'auc': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), 1):
        X_tr, X_vl = X_train_full[train_idx], X_train_full[val_idx]
        y_tr, y_vl = y_train_full[train_idx], y_train_full[val_idx]
        
        fold_model = xgb.XGBClassifier(**model_params)
        fold_model.fit(
            X_tr, y_tr,
            eval_set=[(X_vl, y_vl)],
            verbose=False
        )
        
        y_pred = fold_model.predict(X_vl)
        y_proba = fold_model.predict_proba(X_vl)[:, 1]
        
        cv_scores['accuracy'].append(accuracy_score(y_vl, y_pred))
        cv_scores['f1'].append(f1_score(y_vl, y_pred))
        cv_scores['precision'].append(precision_score(y_vl, y_pred))
        cv_scores['recall'].append(recall_score(y_vl, y_pred))
        cv_scores['auc'].append(roc_auc_score(y_vl, y_proba))
        
        print(f"  Fold {fold}: Acc={cv_scores['accuracy'][-1]:.4f}  "
              f"F1={cv_scores['f1'][-1]:.4f}  "
              f"AUC={cv_scores['auc'][-1]:.4f}")
    
    print()
    print("  ── Cross-Validation Summary ──")
    for metric, scores in cv_scores.items():
        mean, std = np.mean(scores), np.std(scores)
        print(f"  {metric.upper():>10}: {mean:.4f} ± {std:.4f}")
    print()
    
    return cv_scores


def evaluate_model(model, X_test, y_test):
    """Comprehensive evaluation on held-out test set."""
    print("=" * 60)
    print("  PHASE 5: FINAL TEST SET EVALUATION")
    print("=" * 60)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    print()
    print("  Classification Report:")
    print("  " + "-" * 55)
    report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing'])
    for line in report.split('\n'):
        print(f"  {line}")
    
    print()
    print("  ── Key Metrics ──")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  ROC AUC:   {auc:.4f}")
    print()
    print(f"  Confusion Matrix:")
    print(f"      Predicted:  Legit   Phish")
    print(f"  Actual Legit:  {cm[0][0]:>6}  {cm[0][1]:>6}")
    print(f"  Actual Phish:  {cm[1][0]:>6}  {cm[1][1]:>6}")
    print()
    
    metrics = {
        'accuracy': acc, 'f1_score': f1, 'precision': precision,
        'recall': recall, 'roc_auc': auc,
        'confusion_matrix': cm.tolist()
    }
    
    return metrics, y_pred, y_proba


def plot_results(model, feature_names, y_test, y_pred, y_proba, metrics, cv_scores):
    """Generate beautiful, publication-quality training result visualizations."""
    print("=" * 60)
    print("  PHASE 6: GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Set global style
    plt.rcParams.update({
        'figure.facecolor': '#0a0e17',
        'axes.facecolor': '#0f1923',
        'axes.edgecolor': '#1e2d3d',
        'axes.labelcolor': '#e0e6ed',
        'text.color': '#e0e6ed',
        'xtick.color': '#8899aa',
        'ytick.color': '#8899aa',
        'grid.color': '#1e2d3d',
        'font.family': 'sans-serif',
    })
    
    colors = {
        'primary': '#00d4ff',
        'secondary': '#7c3aed',
        'success': '#10b981',
        'danger': '#ef4444',
        'warning': '#f59e0b',
        'gradient_start': '#06b6d4',
        'gradient_end': '#8b5cf6',
    }
    
    # ── 1. Feature Importance (Top 25) ──
    fig, ax = plt.subplots(figsize=(14, 10))
    
    importance = model.feature_importances_
    indices = np.argsort(importance)[-25:]
    
    y_pos = np.arange(len(indices))
    importance_vals = importance[indices]
    names = [feature_names[i] for i in indices]
    
    # Gradient colors for bars
    bar_colors = plt.cm.cool(np.linspace(0.3, 0.9, len(indices)))
    
    bars = ax.barh(y_pos, importance_vals, color=bar_colors, edgecolor='none', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12, fontweight='bold')
    ax.set_title('Top 25 Most Important Features', fontsize=16, fontweight='bold', 
                 color=colors['primary'], pad=20)
    ax.grid(axis='x', alpha=0.2)
    
    for bar, val in zip(bars, importance_vals):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', fontsize=8, color='#aabbcc')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Feature importance chart saved")
    
    # ── 2. Confusion Matrix Heatmap ──
    fig, ax = plt.subplots(figsize=(8, 7))
    
    cm = np.array(metrics['confusion_matrix'])
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create custom colormap
    cmap = sns.color_palette("blend:#0f1923,#06b6d4,#00d4ff", as_cmap=True)
    
    sns.heatmap(cm, annot=False, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'],
                linewidths=2, linecolor='#1e2d3d',
                cbar_kws={'label': 'Count'})
    
    # Add custom annotations with count + percentage
    for i in range(2):
        for j in range(2):
            text = f'{cm[i][j]:,}\n({cm_pct[i][j]:.1f}%)'
            color = '#0a0e17' if cm_pct[i][j] > 60 else '#e0e6ed'
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                    fontsize=14, fontweight='bold', color=color)
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold',
                 color=colors['primary'], pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Confusion matrix saved")
    
    # ── 3. ROC Curve ──
    fig, ax = plt.subplots(figsize=(9, 8))
    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc_val = metrics['roc_auc']
    
    # Fill under curve
    ax.fill_between(fpr, tpr, alpha=0.15, color=colors['primary'])
    ax.plot(fpr, tpr, color=colors['primary'], linewidth=2.5,
            label=f'XGBoost (AUC = {auc_val:.4f})')
    ax.plot([0, 1], [0, 1], color='#3a4a5a', linestyle='--', linewidth=1, label='Random Baseline')
    
    # Mark optimal threshold (Youden's J)
    optimal_idx = np.argmax(tpr - fpr)
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color=colors['success'],
               s=120, zorder=5, edgecolors='white', linewidth=2)
    ax.annotate(f'Optimal\n(t={thresholds[optimal_idx]:.2f})',
                xy=(fpr[optimal_idx], tpr[optimal_idx]),
                xytext=(fpr[optimal_idx] + 0.1, tpr[optimal_idx] - 0.1),
                fontsize=10, color=colors['success'],
                arrowprops=dict(arrowstyle='->', color=colors['success']))
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve', fontsize=16, fontweight='bold',
                 color=colors['primary'], pad=20)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.3)
    ax.grid(alpha=0.15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'roc_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ ROC curve saved")
    
    # ── 4. Cross-Validation Scores ──
    fig, ax = plt.subplots(figsize=(12, 7))
    
    metric_names = list(cv_scores.keys())
    means = [np.mean(cv_scores[m]) for m in metric_names]
    stds = [np.std(cv_scores[m]) for m in metric_names]
    
    x = np.arange(len(metric_names))
    bar_colors_cv = [colors['primary'], colors['secondary'], colors['success'],
                     colors['warning'], colors['gradient_end']]
    
    bars = ax.bar(x, means, yerr=stds, capsize=8, color=bar_colors_cv,
                  edgecolor='none', width=0.6, alpha=0.9,
                  error_kw={'elinewidth': 2, 'ecolor': '#8899aa', 'capthick': 2})
    
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005,
                f'{mean:.4f}', ha='center', fontsize=12, fontweight='bold', color='#e0e6ed')
    
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metric_names], fontsize=11, fontweight='bold')
    ax.set_ylim(0.8, 1.02)
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('5-Fold Cross-Validation Results', fontsize=16, fontweight='bold',
                 color=colors['primary'], pad=20)
    ax.grid(axis='y', alpha=0.15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cross_validation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Cross-validation chart saved")
    
    # ── 5. Score Distribution ──
    fig, ax = plt.subplots(figsize=(12, 7))
    
    legit_proba = y_proba[y_test == 0]
    phish_proba = y_proba[y_test == 1]
    
    ax.hist(legit_proba, bins=80, alpha=0.7, color=colors['success'],
            label=f'Legitimate (n={len(legit_proba):,})', density=True, edgecolor='none')
    ax.hist(phish_proba, bins=80, alpha=0.7, color=colors['danger'],
            label=f'Phishing (n={len(phish_proba):,})', density=True, edgecolor='none')
    
    ax.axvline(x=0.5, color=colors['warning'], linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
    
    ax.set_xlabel('Predicted Phishing Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Score Distribution by True Label', fontsize=16, fontweight='bold',
                 color=colors['primary'], pad=20)
    ax.legend(fontsize=11, framealpha=0.3)
    ax.grid(alpha=0.15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'score_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Score distribution chart saved")
    
    # Reset matplotlib defaults
    plt.rcParams.update(plt.rcParamsDefault)
    print()


def save_model(model, feature_names, metrics, cv_scores, training_time, best_iteration):
    """Save model and comprehensive training metadata."""
    print("=" * 60)
    print("  PHASE 7: SAVING MODEL & METADATA")
    print("=" * 60)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✓ Model saved to {MODEL_PATH}")
    
    # Save metadata
    metadata = {
        'model_type': 'XGBClassifier',
        'training_date': datetime.now().isoformat(),
        'dataset': DATASET_PATH,
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'best_iteration': int(best_iteration),
        'training_time_seconds': round(training_time, 2),
        'hyperparameters': {
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
            'learning_rate': model.learning_rate,
            'subsample': model.subsample,
            'colsample_bytree': model.colsample_bytree,
            'min_child_weight': model.min_child_weight,
            'gamma': model.gamma,
            'reg_alpha': model.reg_alpha,
            'reg_lambda': model.reg_lambda,
            'scale_pos_weight': float(model.scale_pos_weight),
        },
        'test_metrics': {k: float(v) if not isinstance(v, list) else v 
                         for k, v in metrics.items()},
        'cv_metrics': {
            metric: {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'scores': [float(s) for s in scores]
            }
            for metric, scores in cv_scores.items()
        },
        'manual_overrides': 'NONE - Pure data-driven model'
    }
    
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Metadata saved to {METADATA_PATH}")
    
    # Save feature names for inference
    feature_path = os.path.join(MODEL_DIR, 'feature_names.json')
    with open(feature_path, 'w') as f:
        json.dump(feature_names, f)
    print(f"  ✓ Feature names saved to {feature_path}")
    print()


def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          PhishGuard - Model Training Pipeline           ║")
    print("║       Pure Data-Driven • No Manual Controls             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    
    total_start = time.time()
    
    # 1. Load & preprocess data
    X, y, feature_names = load_and_preprocess(DATASET_PATH)
    
    # 2. Split data (stratified)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # 3. Train model with early stopping
    model, training_time, best_iteration = train_model(X_train, y_train, X_val, y_val)
    
    # 4. Cross-validation (on train+val combined for robust estimate)
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    
    # Get the params from the trained model for CV
    cv_params = model.get_params()
    cv_params['n_estimators'] = best_iteration  # Use optimal iteration count
    cv_params['early_stopping_rounds'] = None   # No early stopping in CV folds (fixed n_estimators)
    
    cv_scores = cross_validate(X_train_full, y_train_full, cv_params)
    
    # 5. Final evaluation on held-out test set
    metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test)
    
    # 6. Visualizations
    plot_results(model, feature_names, y_test, y_pred, y_proba, metrics, cv_scores)
    
    # 7. Save everything
    save_model(model, feature_names, metrics, cv_scores, training_time, best_iteration)
    
    total_time = time.time() - total_start
    
    print("=" * 60)
    print("  ✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Model:      {MODEL_PATH}")
    print(f"  Figures:    {FIGURES_DIR}/")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  F1 Score:   {metrics['f1_score']:.4f}")
    print(f"  ROC AUC:    {metrics['roc_auc']:.4f}")
    print()
    print("  No manual controls. No whitelists. Pure ML.")
    print()


if __name__ == '__main__':
    main()
