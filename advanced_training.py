"""
Advanced Heart Disease Prediction Training Pipeline
====================================================

Integrates 10 advanced ML techniques:
1.  BorutaSHAP Feature Selection
2.  mRMR (Minimum Redundancy Maximum Relevance)
3.  Improved Weighted Random Forest (IWRF)
4.  LightGBM (Light Gradient Boosting Machine)
5.  CatBoost
6.  EasyEnsemble
7.  Bayesian Optimization
8.  Cost-Sensitive Learning
9.  Probability Calibration
10. Threshold Optimization
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    auc,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.base import BaseEstimator, ClassifierMixin

import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.ensemble import EasyEnsembleClassifier

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================================
# 1. DATA LOADING & EXPLORATION
# ============================================================================

def load_and_explore(csv_path: str):
    """Load dataset and print summary statistics."""
    print("=" * 70)
    print("STAGE 1: DATA LOADING & EXPLORATION")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\n  Target distribution:")
    counts = df["heart_disease"].value_counts()
    for label, cnt in counts.items():
        print(f"    Class {label}: {cnt}  ({cnt / len(df):.1%})")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    return df


# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply domain-driven feature engineering."""
    print("\n" + "=" * 70)
    print("STAGE 2: FEATURE ENGINEERING")
    print("=" * 70)

    df_feat = df.copy()

    # Age groups
    df_feat["age_group"] = pd.cut(
        df_feat["age"], bins=[0, 40, 55, 70, 100], labels=[0, 1, 2, 3]
    ).astype(int)

    # BMI categories
    df_feat["bmi_category"] = pd.cut(
        df_feat["bmi"], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]
    ).astype(int)

    # Blood pressure categories
    df_feat["bp_category"] = pd.cut(
        df_feat["trestbps"], bins=[0, 120, 140, 180, 300], labels=[0, 1, 2, 3]
    ).astype(int)

    # Cholesterol categories
    df_feat["chol_category"] = pd.cut(
        df_feat["chol"], bins=[0, 200, 240, 1000], labels=[0, 1, 2]
    ).astype(int)

    # Heart rate categories
    df_feat["hr_category"] = pd.cut(
        df_feat["thalach"], bins=[0, 100, 150, 220], labels=[0, 1, 2]
    ).astype(int)

    # Risk combination score
    df_feat["risk_score"] = (
        df_feat["smoking"]
        + df_feat["diabetes"]
        + (df_feat["age"] > 55).astype(int)
        + (df_feat["sex"] == 1).astype(int)
    )

    # Interaction features
    df_feat["age_sex"] = df_feat["age"] * df_feat["sex"]
    df_feat["chol_age"] = df_feat["chol"] * df_feat["age"]
    df_feat["bp_age"] = df_feat["trestbps"] * df_feat["age"]
    df_feat["chol_bp"] = df_feat["chol"] * df_feat["trestbps"]
    df_feat["hr_age"] = df_feat["thalach"] * df_feat["age"]
    df_feat["oldpeak_slope"] = df_feat["oldpeak"] * df_feat["slope"]

    n_features = len(df_feat.columns) - 1  # exclude target
    print(f"  Engineered feature count: {n_features}")
    return df_feat


# ============================================================================
# 3. BorutaSHAP FEATURE SELECTION
# ============================================================================

def boruta_shap_selection(X_train, y_train, feature_names):
    """
    BorutaSHAP-inspired feature selection using SHAP values with
    a shadow-feature permutation test (Boruta algorithm).
    """
    print("\n" + "=" * 70)
    print("STAGE 3: BorutaSHAP FEATURE SELECTION")
    print("=" * 70)

    import shap

    # Train a LightGBM model for SHAP computation
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        random_state=42, verbose=-1, n_jobs=-1,
    )
    lgb_model.fit(X_train, y_train)

    # Compute SHAP values
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_train)

    # Handle multi-output SHAP (binary classification)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Boruta-style shadow test: create shadow features and compare
    n_trials = 20
    hit_counts = np.zeros(len(feature_names))

    for trial in range(n_trials):
        # Create shadow features (permuted copies)
        X_shadow = X_train.copy()
        for col_idx in range(X_shadow.shape[1]):
            X_shadow[:, col_idx] = np.random.permutation(X_shadow[:, col_idx])

        X_combined = np.hstack([X_train, X_shadow])

        lgb_shadow = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.05, num_leaves=31,
            random_state=42 + trial, verbose=-1, n_jobs=-1,
        )
        lgb_shadow.fit(X_combined, y_train)

        explainer_s = shap.TreeExplainer(lgb_shadow)
        shap_vals_s = explainer_s.shap_values(X_combined)
        if isinstance(shap_vals_s, list):
            shap_vals_s = shap_vals_s[1]

        abs_shap_combined = np.abs(shap_vals_s).mean(axis=0)
        real_shap = abs_shap_combined[: len(feature_names)]
        shadow_max = abs_shap_combined[len(feature_names) :].max()

        hits = real_shap > shadow_max
        hit_counts += hits.astype(int)

    # Features confirmed in >= 50% of trials
    confirmed_mask = hit_counts >= (n_trials * 0.5)
    selected = [f for f, m in zip(feature_names, confirmed_mask) if m]

    # Always keep at least top-8 by raw SHAP importance as fallback
    if len(selected) < 8:
        top_idx = np.argsort(mean_abs_shap)[::-1][:8]
        selected = [feature_names[i] for i in top_idx]

    print(f"  BorutaSHAP selected {len(selected)} features:")
    for f in selected:
        idx = feature_names.index(f)
        print(f"    - {f}  (mean |SHAP| = {mean_abs_shap[idx]:.4f})")

    return selected, mean_abs_shap


# ============================================================================
# 4. mRMR FEATURE SELECTION
# ============================================================================

def mrmr_selection(X_train_df, y_train, n_select=12):
    """
    Minimum Redundancy Maximum Relevance feature selection.
    Uses mutual-information-based scoring.
    """
    print("\n" + "=" * 70)
    print("STAGE 4: mRMR FEATURE SELECTION")
    print("=" * 70)

    try:
        from mrmr import mrmr_classif
        df_temp = X_train_df.copy()
        df_temp["target"] = y_train.values if hasattr(y_train, "values") else y_train
        selected = mrmr_classif(
            X=df_temp.drop("target", axis=1),
            y=df_temp["target"],
            K=min(n_select, len(X_train_df.columns)),
        )
        print(f"  mRMR selected {len(selected)} features: {selected}")
        return selected
    except ImportError:
        print("  [WARN] mrmr-selection not available, using MI-based fallback")
        from sklearn.feature_selection import mutual_info_classif

        mi = mutual_info_classif(X_train_df, y_train, random_state=42)
        mi_series = pd.Series(mi, index=X_train_df.columns).sort_values(ascending=False)

        # Greedy mRMR approximation
        selected = [mi_series.index[0]]
        remaining = list(mi_series.index[1:])

        for _ in range(min(n_select - 1, len(remaining))):
            best_score = -np.inf
            best_feat = None
            for feat in remaining:
                relevance = mi_series[feat]
                redundancy = X_train_df[selected].corrwith(X_train_df[feat]).abs().mean()
                score = relevance - redundancy
                if score > best_score:
                    best_score = score
                    best_feat = feat
            if best_feat is not None:
                selected.append(best_feat)
                remaining.remove(best_feat)

        print(f"  mRMR (fallback) selected {len(selected)} features: {selected}")
        return selected


def combine_feature_selections(boruta_features, mrmr_features, all_features):
    """Combine BorutaSHAP and mRMR results via union."""
    combined = list(set(boruta_features) | set(mrmr_features))
    # Preserve original order
    combined_ordered = [f for f in all_features if f in combined]
    print(f"\n  Combined feature set ({len(combined_ordered)} features): {combined_ordered}")
    return combined_ordered


# ============================================================================
# 5. IMPROVED WEIGHTED RANDOM FOREST (IWRF)
# ============================================================================

class ImprovedWeightedRandomForest(BaseEstimator, ClassifierMixin):
    """
    Improved Weighted Random Forest (IWRF):
    - Dynamic sample weighting based on class imbalance
    - Feature importance feedback loop for iterative refinement
    - Cost-sensitive splits via class_weight='balanced_subsample'
    """

    def __init__(self, n_estimators=300, max_depth=None, min_samples_split=5,
                 min_samples_leaf=2, max_features="sqrt", n_iterations=2,
                 random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.classes_ = None
        self.final_model_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        sample_weights = self._compute_class_weights(y)

        for iteration in range(self.n_iterations):
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                class_weight="balanced_subsample",
                random_state=self.random_state + iteration,
                n_jobs=-1,
            )
            rf.fit(X, y, sample_weight=sample_weights)

            # Update sample weights using OOB-style misclassification feedback
            preds = rf.predict(X)
            misclassified = (preds != y).astype(float)
            # Increase weight for hard samples
            boost_factor = 1.0 + 0.5 * misclassified
            sample_weights = sample_weights * boost_factor
            sample_weights = sample_weights / sample_weights.sum() * len(y)

        self.final_model_ = rf
        self.feature_importances_ = rf.feature_importances_
        return self

    def predict(self, X):
        return self.final_model_.predict(X)

    def predict_proba(self, X):
        return self.final_model_.predict_proba(X)

    def _compute_class_weights(self, y):
        """Compute inverse-frequency sample weights."""
        classes, counts = np.unique(y, return_counts=True)
        weight_map = {c: len(y) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        return np.array([weight_map[label] for label in y])


# ============================================================================
# 6-8. MODEL BUILDERS (LightGBM, CatBoost, EasyEnsemble)
# ============================================================================

def build_lightgbm(random_state=42):
    """LightGBM with cost-sensitive is_unbalance."""
    return lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        is_unbalance=True,  # Cost-sensitive learning
        random_state=random_state,
        verbose=-1,
        n_jobs=-1,
    )


def build_catboost(random_state=42):
    """CatBoost with auto class weight balancing."""
    return CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        auto_class_weights="Balanced",  # Cost-sensitive learning
        random_seed=random_state,
        verbose=0,
        eval_metric="F1",
    )


def build_easy_ensemble(random_state=42):
    """EasyEnsemble: ensemble of balanced bagged classifiers."""
    return EasyEnsembleClassifier(
        n_estimators=15,
        random_state=random_state,
        n_jobs=-1,
    )


# ============================================================================
# 9. BAYESIAN OPTIMIZATION
# ============================================================================

def bayesian_optimize_models(X_train, y_train, selected_features):
    """
    Bayesian optimization of hyperparameters for all base models
    using scikit-optimize BayesSearchCV.
    """
    print("\n" + "=" * 70)
    print("STAGE 9: BAYESIAN OPTIMIZATION")
    print("=" * 70)

    from skopt import BayesSearchCV
    from skopt.space import Integer, Real, Categorical

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    optimized_models = {}

    # --- IWRF ---
    print("\n  [1/4] Optimizing IWRF...")
    iwrf_space = {
        "n_estimators": Integer(100, 500),
        "max_depth": Integer(5, 20),
        "min_samples_split": Integer(2, 15),
        "min_samples_leaf": Integer(1, 8),
        "max_features": Categorical(["sqrt", "log2"]),
    }
    iwrf_opt = BayesSearchCV(
        ImprovedWeightedRandomForest(random_state=42),
        iwrf_space, n_iter=20, cv=cv, scoring="f1",
        random_state=42, n_jobs=-1, verbose=0,
    )
    iwrf_opt.fit(X_train, y_train)
    optimized_models["iwrf"] = iwrf_opt.best_estimator_
    print(f"    Best F1: {iwrf_opt.best_score_:.4f}")
    print(f"    Best params: {iwrf_opt.best_params_}")

    # --- LightGBM ---
    print("\n  [2/4] Optimizing LightGBM...")
    lgbm_space = {
        "n_estimators": Integer(200, 800),
        "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
        "num_leaves": Integer(15, 63),
        "max_depth": Integer(4, 10),
        "min_child_samples": Integer(10, 50),
        "subsample": Real(0.6, 1.0),
        "colsample_bytree": Real(0.6, 1.0),
        "reg_alpha": Real(1e-3, 10, prior="log-uniform"),
        "reg_lambda": Real(1e-3, 10, prior="log-uniform"),
    }
    lgbm_base = lgb.LGBMClassifier(
        is_unbalance=True, random_state=42, verbose=-1, n_jobs=1,
    )
    lgbm_opt = BayesSearchCV(
        lgbm_base, lgbm_space, n_iter=25, cv=cv, scoring="f1",
        random_state=42, n_jobs=-1, verbose=0,
    )
    lgbm_opt.fit(X_train, y_train)
    optimized_models["lgbm"] = lgbm_opt.best_estimator_
    print(f"    Best F1: {lgbm_opt.best_score_:.4f}")
    print(f"    Best params: {lgbm_opt.best_params_}")

    # --- CatBoost ---
    print("\n  [3/4] Optimizing CatBoost...")
    catboost_space = {
        "iterations": Integer(200, 800),
        "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
        "depth": Integer(4, 10),
        "l2_leaf_reg": Real(1e-2, 10, prior="log-uniform"),
    }
    cb_base = CatBoostClassifier(
        auto_class_weights="Balanced", random_seed=42, verbose=0,
        eval_metric="F1",
    )
    cb_opt = BayesSearchCV(
        cb_base, catboost_space, n_iter=20, cv=cv, scoring="f1",
        random_state=42, n_jobs=-1, verbose=0,
    )
    cb_opt.fit(X_train, y_train)
    optimized_models["catboost"] = cb_opt.best_estimator_
    print(f"    Best F1: {cb_opt.best_score_:.4f}")
    print(f"    Best params: {cb_opt.best_params_}")

    # --- EasyEnsemble ---
    print("\n  [4/4] Optimizing EasyEnsemble...")
    ee_space = {
        "n_estimators": Integer(5, 30),
    }
    ee_base = EasyEnsembleClassifier(random_state=42, n_jobs=1)
    ee_opt = BayesSearchCV(
        ee_base, ee_space, n_iter=10, cv=cv, scoring="f1",
        random_state=42, n_jobs=-1, verbose=0,
    )
    ee_opt.fit(X_train, y_train)
    optimized_models["easy_ensemble"] = ee_opt.best_estimator_
    print(f"    Best F1: {ee_opt.best_score_:.4f}")
    print(f"    Best params: {ee_opt.best_params_}")

    return optimized_models


# ============================================================================
# 10. COST-SENSITIVE EVALUATION
# ============================================================================

def cost_sensitive_evaluate(y_true, y_pred, y_prob):
    """
    Evaluate predictions with a cost matrix:
      - False Negative (missed disease) costs 5×
      - False Positive (false alarm) costs 1×
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    cost_fn = 5  # Missing disease is very costly
    cost_fp = 1  # False alarm is less costly
    total_cost = fn * cost_fn + fp * cost_fp

    print(f"\n  Cost-Sensitive Evaluation:")
    print(f"    TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"    Total cost (FN×{cost_fn} + FP×{cost_fp}): {total_cost}")
    print(f"    Cost per sample: {total_cost / len(y_true):.4f}")

    return total_cost


# ============================================================================
# 11. ENSEMBLE CONSTRUCTION
# ============================================================================

def build_voting_ensemble(optimized_models):
    """Build a soft-voting ensemble from the optimized base models."""
    print("\n" + "=" * 70)
    print("STAGE 11: SOFT VOTING ENSEMBLE")
    print("=" * 70)

    estimators = [
        ("iwrf", optimized_models["iwrf"]),
        ("lgbm", optimized_models["lgbm"]),
        ("catboost", optimized_models["catboost"]),
        ("easy_ensemble", optimized_models["easy_ensemble"]),
    ]

    # VotingClassifier requires re-fitting. We use pre-fitted models via a wrapper.
    ensemble = PreFittedVotingClassifier(estimators=estimators)
    print("  Ensemble created with 4 base models (soft voting)")
    return ensemble


class PreFittedVotingClassifier(BaseEstimator, ClassifierMixin):
    """
    Soft voting classifier that uses already-fitted estimators.
    Avoids re-training inside VotingClassifier.
    """

    def __init__(self, estimators=None, weights=None):
        self.estimators = estimators or []
        self.weights = weights
        self.classes_ = None
        self.estimators_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.estimators_ = [(name, est) for name, est in self.estimators]
        # Compute validation F1 for weighting
        scores = []
        for name, est in self.estimators:
            pred = est.predict(X)
            f1 = f1_score(y, pred)
            scores.append(f1)
            print(f"    {name:20s}  train F1 = {f1:.4f}")
        # Weight by F1 score
        total = sum(scores)
        self.weights = [s / total for s in scores]
        print(f"    Ensemble weights: {[f'{w:.3f}' for w in self.weights]}")
        return self

    def predict_proba(self, X):
        probas = []
        for (name, est), w in zip(self.estimators_, self.weights):
            p = est.predict_proba(X)
            probas.append(w * p)
        avg_proba = np.sum(probas, axis=0)
        return avg_proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ============================================================================
# 12. PROBABILITY CALIBRATION
# ============================================================================

def calibrate_model(ensemble, X_cal, y_cal):
    """
    Apply probability calibration using isotonic regression.
    Produces well-calibrated probability estimates.
    """
    print("\n" + "=" * 70)
    print("STAGE 12: PROBABILITY CALIBRATION")
    print("=" * 70)

    calibrated = CalibratedClassifierCV(
        ensemble, method="isotonic", cv="prefit"
    )
    calibrated.fit(X_cal, y_cal)

    # Evaluate calibration
    prob_before = ensemble.predict_proba(X_cal)[:, 1]
    prob_after = calibrated.predict_proba(X_cal)[:, 1]

    brier_before = brier_score_loss(y_cal, prob_before)
    brier_after = brier_score_loss(y_cal, prob_after)

    print(f"  Brier score before calibration: {brier_before:.4f}")
    print(f"  Brier score after calibration:  {brier_after:.4f}")
    print(f"  Improvement: {(brier_before - brier_after) / brier_before:.1%}")

    return calibrated


# ============================================================================
# 13. THRESHOLD OPTIMIZATION
# ============================================================================

def optimize_threshold(model, X_val, y_val):
    """
    Find the optimal decision threshold by maximizing:
      - Youden's J statistic (Sensitivity + Specificity - 1)
      - Also report threshold for best F1
    """
    print("\n" + "=" * 70)
    print("STAGE 13: THRESHOLD OPTIMIZATION")
    print("=" * 70)

    y_proba = model.predict_proba(X_val)[:, 1]

    # Youden's J
    fpr, tpr, roc_thresholds = roc_curve(y_val, y_proba)
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    best_threshold_j = roc_thresholds[best_j_idx]

    # Best F1
    precisions, recalls, pr_thresholds = precision_recall_curve(y_val, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold_f1 = pr_thresholds[min(best_f1_idx, len(pr_thresholds) - 1)]

    # Use Youden's J as primary (better for medical screening — favors sensitivity)
    optimal_threshold = float(best_threshold_j)

    print(f"  Default threshold (0.5):")
    default_pred = (y_proba >= 0.5).astype(int)
    print(f"    F1={f1_score(y_val, default_pred):.4f}  "
          f"Acc={accuracy_score(y_val, default_pred):.4f}  "
          f"Sens={recall_score(y_val, default_pred):.4f}  "
          f"Prec={precision_score(y_val, default_pred):.4f}")

    print(f"\n  Youden's J optimal threshold: {best_threshold_j:.4f}")
    opt_pred_j = (y_proba >= best_threshold_j).astype(int)
    print(f"    F1={f1_score(y_val, opt_pred_j):.4f}  "
          f"Acc={accuracy_score(y_val, opt_pred_j):.4f}  "
          f"Sens={recall_score(y_val, opt_pred_j):.4f}  "
          f"Prec={precision_score(y_val, opt_pred_j):.4f}")

    print(f"\n  Best F1 threshold: {best_threshold_f1:.4f}")
    opt_pred_f1 = (y_proba >= best_threshold_f1).astype(int)
    print(f"    F1={f1_score(y_val, opt_pred_f1):.4f}  "
          f"Acc={accuracy_score(y_val, opt_pred_f1):.4f}  "
          f"Sens={recall_score(y_val, opt_pred_f1):.4f}  "
          f"Prec={precision_score(y_val, opt_pred_f1):.4f}")

    print(f"\n  >> Selected threshold: {optimal_threshold:.4f} (Youden's J)")

    return optimal_threshold


# ============================================================================
# 14. EVALUATION & VISUALIZATION
# ============================================================================

def full_evaluation(model, X_test, y_test, threshold, output_dir="."):
    """Comprehensive evaluation with plots."""
    print("\n" + "=" * 70)
    print("STAGE 14: FINAL EVALUATION")
    print("=" * 70)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    # PR-AUC
    pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(pr_rec, pr_prec)

    print(f"\n  {'Metric':<25} {'Value':>10}")
    print(f"  {'-'*35}")
    print(f"  {'Accuracy':<25} {acc:>10.4f}")
    print(f"  {'F1 Score':<25} {f1:>10.4f}")
    print(f"  {'Precision':<25} {prec:>10.4f}")
    print(f"  {'Recall (Sensitivity)':<25} {rec:>10.4f}")
    print(f"  {'ROC-AUC':<25} {roc:>10.4f}")
    print(f"  {'PR-AUC':<25} {pr_auc:>10.4f}")
    print(f"  {'Brier Score':<25} {brier:>10.4f}")
    print(f"  {'Threshold used':<25} {threshold:>10.4f}")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

    # Cost-sensitive evaluation
    total_cost = cost_sensitive_evaluate(y_test, y_pred, y_proba)

    # --- Confusion Matrix Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0],
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    axes[0, 0].set_title("Confusion Matrix", fontsize=14)
    axes[0, 0].set_ylabel("Actual")
    axes[0, 0].set_xlabel("Predicted")

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0, 1].plot(fpr, tpr, "b-", linewidth=2, label=f"ROC-AUC = {roc:.4f}")
    axes[0, 1].plot([0, 1], [0, 1], "r--", alpha=0.5)
    axes[0, 1].fill_between(fpr, tpr, alpha=0.1, color="blue")
    axes[0, 1].set_xlabel("False Positive Rate")
    axes[0, 1].set_ylabel("True Positive Rate")
    axes[0, 1].set_title("ROC Curve", fontsize=14)
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(True, alpha=0.3)

    # --- Precision-Recall Curve ---
    axes[1, 0].plot(pr_rec, pr_prec, "g-", linewidth=2, label=f"PR-AUC = {pr_auc:.4f}")
    axes[1, 0].fill_between(pr_rec, pr_prec, alpha=0.1, color="green")
    axes[1, 0].set_xlabel("Recall")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_title("Precision-Recall Curve", fontsize=14)
    axes[1, 0].legend(loc="lower left")
    axes[1, 0].grid(True, alpha=0.3)

    # --- Calibration Curve ---
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
    axes[1, 1].plot(prob_pred, prob_true, "o-", color="purple", linewidth=2,
                     label="Calibrated model")
    axes[1, 1].plot([0, 1], [0, 1], "r--", alpha=0.5, label="Perfect calibration")
    axes[1, 1].set_xlabel("Mean Predicted Probability")
    axes[1, 1].set_ylabel("Fraction of Positives")
    axes[1, 1].set_title("Calibration Curve", fontsize=14)
    axes[1, 1].legend(loc="lower right")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "evaluation_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Evaluation plots saved to: {plot_path}")

    metrics = {
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall_sensitivity": round(rec, 4),
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "optimal_threshold": round(threshold, 4),
        "total_cost": int(total_cost),
    }
    return metrics


# ============================================================================
# CUSTOM FEATURE SELECTOR (for pipeline serialization)
# ============================================================================

class FeatureSelector(BaseEstimator):
    """Selectable feature mask that can be saved/loaded with joblib."""

    def __init__(self, selected_features=None, all_features=None):
        self.selected_features = selected_features or []
        self.all_features = all_features or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features].values
        else:
            indices = [self.all_features.index(f) for f in self.selected_features]
            return X[:, indices]

    def get_feature_names_out(self):
        return self.selected_features


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("\n" + "#" * 70)
    print("#  ADVANCED HEART DISEASE PREDICTION TRAINING PIPELINE")
    print("#  Techniques: IWRF | LightGBM | CatBoost | BorutaSHAP | mRMR")
    print("#  EasyEnsemble | Bayesian Opt | Cost-Sensitive | Calibration")
    print("#  Threshold Optimization")
    print("#" * 70)

    csv_path = "heart_disease_dataset.csv"
    output_dir = "."

    # ---- Stage 1: Load data ----
    df = load_and_explore(csv_path)

    # ---- Stage 2: Feature engineering ----
    df_feat = create_features(df)

    X = df_feat.drop("heart_disease", axis=1)
    y = df_feat["heart_disease"]
    feature_names = list(X.columns)

    # Split: 70% train, 15% calibration, 15% test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_full, y_train_full, test_size=0.176,  # 0.176 of 0.85 ≈ 0.15
        stratify=y_train_full, random_state=42
    )

    print(f"\n  Train: {X_train.shape[0]}, Calibration: {X_cal.shape[0]}, Test: {X_test.shape[0]}")

    # ---- Stage 3: BorutaSHAP ----
    boruta_features, shap_importances = boruta_shap_selection(
        X_train.values, y_train.values, feature_names
    )

    # ---- Stage 4: mRMR ----
    mrmr_features = mrmr_selection(X_train, y_train, n_select=12)

    # ---- Combine feature selections ----
    selected_features = combine_feature_selections(
        boruta_features, mrmr_features, feature_names
    )

    # Create feature selector object
    selector = FeatureSelector(
        selected_features=selected_features, all_features=feature_names
    )

    # Filter data to selected features
    X_train_sel = X_train[selected_features].values
    X_cal_sel = X_cal[selected_features].values
    X_test_sel = X_test[selected_features].values

    # ---- Scale features ----
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_cal_scaled = scaler.transform(X_cal_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    # ---- Stage 5-8: Build base models ----
    print("\n" + "=" * 70)
    print("STAGES 5-8: TRAINING BASE MODELS (before optimization)")
    print("=" * 70)

    print("\n  Training IWRF...")
    iwrf = ImprovedWeightedRandomForest(n_estimators=300, random_state=42)
    iwrf.fit(X_train_scaled, y_train.values)
    print(f"    Train accuracy: {accuracy_score(y_train, iwrf.predict(X_train_scaled)):.4f}")

    print("\n  Training LightGBM...")
    lgbm = build_lightgbm()
    lgbm.fit(X_train_scaled, y_train.values)
    print(f"    Train accuracy: {accuracy_score(y_train, lgbm.predict(X_train_scaled)):.4f}")

    print("\n  Training CatBoost...")
    cb = build_catboost()
    cb.fit(X_train_scaled, y_train.values)
    print(f"    Train accuracy: {accuracy_score(y_train, cb.predict(X_train_scaled)):.4f}")

    print("\n  Training EasyEnsemble...")
    ee = build_easy_ensemble()
    ee.fit(X_train_scaled, y_train.values)
    print(f"    Train accuracy: {accuracy_score(y_train, ee.predict(X_train_scaled)):.4f}")

    # ---- Stage 9: Bayesian Optimization ----
    optimized_models = bayesian_optimize_models(
        X_train_scaled, y_train.values, selected_features
    )

    # ---- Stage 10: Cost-sensitive evaluation of individual models ----
    print("\n" + "=" * 70)
    print("STAGE 10: COST-SENSITIVE LEARNING – Individual Model Evaluation")
    print("=" * 70)
    for name, model in optimized_models.items():
        pred = model.predict(X_test_scaled)
        prob = model.predict_proba(X_test_scaled)[:, 1]
        print(f"\n  Model: {name}")
        print(f"    Accuracy:  {accuracy_score(y_test, pred):.4f}")
        print(f"    F1:        {f1_score(y_test, pred):.4f}")
        print(f"    ROC-AUC:   {roc_auc_score(y_test, prob):.4f}")
        cost_sensitive_evaluate(y_test, pred, prob)

    # ---- Stage 11: Ensemble ----
    ensemble = build_voting_ensemble(optimized_models)
    ensemble.fit(X_train_scaled, y_train.values)

    # ---- Stage 12: Probability Calibration ----
    calibrated_ensemble = calibrate_model(ensemble, X_cal_scaled, y_cal.values)

    # ---- Stage 13: Threshold Optimization ----
    optimal_threshold = optimize_threshold(calibrated_ensemble, X_cal_scaled, y_cal.values)

    # ---- Stage 14: Final Evaluation ----
    metrics = full_evaluation(
        calibrated_ensemble, X_test_scaled, y_test.values,
        optimal_threshold, output_dir
    )

    # ---- Save Artifacts ----
    print("\n" + "=" * 70)
    print("SAVING ARTIFACTS")
    print("=" * 70)

    joblib.dump(calibrated_ensemble, os.path.join(output_dir, "optimized_heart_disease_model.pkl"))
    print("  Saved: optimized_heart_disease_model.pkl")

    joblib.dump(scaler, os.path.join(output_dir, "robust_scaler.pkl"))
    print("  Saved: robust_scaler.pkl")

    joblib.dump(selector, os.path.join(output_dir, "feature_selector.pkl"))
    print("  Saved: feature_selector.pkl")

    joblib.dump(optimal_threshold, os.path.join(output_dir, "optimal_threshold.pkl"))
    print("  Saved: optimal_threshold.pkl")

    # Save metrics
    metrics["techniques_used"] = [
        "Improved Weighted Random Forest (IWRF)",
        "LightGBM",
        "CatBoost",
        "EasyEnsemble",
        "BorutaSHAP Feature Selection",
        "mRMR Feature Selection",
        "Bayesian Optimization",
        "Cost-Sensitive Learning",
        "Probability Calibration (Isotonic)",
        "Threshold Optimization (Youden's J)",
    ]
    metrics["selected_features"] = selected_features
    metrics["n_train_samples"] = int(X_train.shape[0])
    metrics["n_test_samples"] = int(X_test.shape[0])

    with open(os.path.join(output_dir, "training_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("  Saved: training_results.json")

    print("\n" + "#" * 70)
    print("#  TRAINING COMPLETE!")
    print(f"#  Final Test Accuracy:  {metrics['accuracy']}")
    print(f"#  Final Test F1 Score:  {metrics['f1_score']}")
    print(f"#  Final Test ROC-AUC:   {metrics['roc_auc']}")
    print(f"#  Optimal Threshold:    {metrics['optimal_threshold']}")
    print("#" * 70)


if __name__ == "__main__":
    main()
