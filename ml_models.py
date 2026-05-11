import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
import json
import numpy as np

MODEL_DIR = 'trained_model'
MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_phishguard.pkl')
METADATA_PATH = os.path.join(MODEL_DIR, 'training_metadata.json')

class MLModels:
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        self.metadata = None

    def train(self, X, y):
        """Train from scratch (used by train.py or legacy code)."""
        print(f"Training {self.model_type} model...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if self.model_type == 'xgboost':
            # Calculate class imbalance
            neg = (y_train == 0).sum()
            pos = (y_train == 1).sum()
            
            self.model = xgb.XGBClassifier(
                n_estimators=1000,
                max_depth=7,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_weight=5,
                gamma=0.2,
                reg_alpha=0.1,
                reg_lambda=1.5,
                scale_pos_weight=neg/pos,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                tree_method='hist',
                early_stopping_rounds=50,
            )
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
        
        y_pred = self.model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        return accuracy_score(y_test, y_pred)

    def load_trained(self):
        """Load pre-trained model from trained_model/ directory."""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. "
                f"Run 'python train.py' first to train the model."
            )
        
        with open(MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)
        
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                self.metadata = json.load(f)
            print(f"[OK] Loaded model trained on {self.metadata.get('training_date', 'unknown')}")
            print(f"  Test accuracy: {self.metadata['test_metrics']['accuracy']:.4f}")
            print(f"  ROC AUC:      {self.metadata['test_metrics']['roc_auc']:.4f}")
        else:
            print("[OK] Model loaded (no metadata found)")
        
        return self

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
