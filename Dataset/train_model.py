"""
DUI Detection ML Model Training Script
Uses Random Forest + Gradient Boosting ensemble for high accuracy
"""
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score)
from sklearn.svm import SVC

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'Dataset')


def generate_synthetic_dataset(n_samples=2000, save=True):
    """
    Generate a realistic synthetic DUI detection dataset.
    Features based on real-world DUI detection research.
    """
    np.random.seed(42)
    n_sober = n_samples // 2
    n_dui = n_samples // 2

    # --- SOBER drivers ---
    sober = {
        'reaction_time':     np.random.normal(0.8, 0.15, n_sober).clip(0.4, 1.3),
        'steering_deviation': np.random.normal(5, 2, n_sober).clip(0, 12),
        'speed_variation':   np.random.normal(8, 3, n_sober).clip(2, 18),
        'lane_deviation':    np.random.normal(0.15, 0.05, n_sober).clip(0, 0.35),
        'brake_pressure':    np.random.normal(40, 10, n_sober).clip(20, 65),
        'acceleration_jerk': np.random.normal(1.5, 0.5, n_sober).clip(0.3, 2.8),
        'eye_blink_rate':    np.random.normal(17, 3, n_sober).clip(10, 25),
        'head_tilt_angle':   np.random.normal(5, 2, n_sober).clip(0, 12),
        'heart_rate':        np.random.normal(72, 8, n_sober).clip(55, 90),
        'skin_conductance':  np.random.normal(3.0, 0.8, n_sober).clip(1.0, 5.0),
        'label': np.zeros(n_sober, dtype=int)
    }

    # --- DUI drivers ---
    dui = {
        'reaction_time':     np.random.normal(1.9, 0.35, n_dui).clip(1.2, 3.5),
        'steering_deviation': np.random.normal(22, 6, n_dui).clip(12, 45),
        'speed_variation':   np.random.normal(28, 8, n_dui).clip(15, 55),
        'lane_deviation':    np.random.normal(0.55, 0.15, n_dui).clip(0.3, 1.2),
        'brake_pressure':    np.random.normal(68, 15, n_dui).clip(40, 100),
        'acceleration_jerk': np.random.normal(4.5, 1.2, n_dui).clip(2.5, 8.0),
        'eye_blink_rate':    np.random.normal(9, 3, n_dui).clip(2, 16),
        'head_tilt_angle':   np.random.normal(22, 6, n_dui).clip(12, 45),
        'heart_rate':        np.random.normal(90, 12, n_dui).clip(70, 130),
        'skin_conductance':  np.random.normal(6.5, 1.5, n_dui).clip(4.0, 12.0),
        'label': np.ones(n_dui, dtype=int)
    }

    df_sober = pd.DataFrame(sober)
    df_dui = pd.DataFrame(dui)
    df = pd.concat([df_sober, df_dui], ignore_index=True).sample(frac=1, random_state=42)

    if save:
        os.makedirs(DATASET_DIR, exist_ok=True)
        df.to_csv(os.path.join(DATASET_DIR, 'dui_dataset.csv'), index=False)
        print(f"[✓] Dataset saved: {len(df)} samples ({n_sober} sober, {n_dui} DUI)")

    return df


def train_dui_model():
    """Train and save the DUI detection model."""
    csv_path = os.path.join(DATASET_DIR, 'dui_dataset.csv')

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"[✓] Loaded existing dataset: {len(df)} samples")
    else:
        print("[*] Generating synthetic dataset...")
        df = generate_synthetic_dataset()

    feature_cols = [
        'reaction_time', 'steering_deviation', 'speed_variation',
        'lane_deviation', 'brake_pressure', 'acceleration_jerk',
        'eye_blink_rate', 'head_tilt_angle', 'heart_rate', 'skin_conductance'
    ]

    X = df[feature_cols].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build ensemble model
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
    svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)

    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('svm', svm)],
        voting='soft'
    )

    print("[*] Training ensemble model (RF + GB + SVM)...")
    ensemble.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = ensemble.predict(X_test_scaled)
    y_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cv_scores = cross_val_score(ensemble, X_train_scaled, y_train, cv=5, scoring='accuracy')

    print(f"\n{'='*50}")
    print(f"  MODEL EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  Accuracy      : {acc*100:.2f}%")
    print(f"  ROC-AUC Score : {auc:.4f}")
    print(f"  CV Score (5-fold): {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Sober','DUI'])}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Save model and scaler
    model_path = os.path.join(DATASET_DIR, 'dui_model.pkl')
    scaler_path = os.path.join(DATASET_DIR, 'scaler.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(ensemble, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Save metrics for display
    metrics = {
        'accuracy': round(acc * 100, 2),
        'auc': round(auc, 4),
        'cv_mean': round(cv_scores.mean() * 100, 2),
        'cv_std': round(cv_scores.std() * 100, 2),
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred,
                                   target_names=['Sober', 'DUI'], output_dict=True)
    }

    metrics_path = os.path.join(DATASET_DIR, 'model_metrics.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)

    print(f"\n[✓] Model saved to: {model_path}")
    print(f"[✓] Scaler saved to: {scaler_path}")
    return metrics


if __name__ == '__main__':
    train_dui_model()
