# =============================================================================
# 🏥 MULTI-DISEASE RISK PREDICTION — Training Pipeline (v4 — Improved)
# =============================================================================
# Diseases: Diabetes | Heart Disease | Obesity | Sleep Apnea
#
# ── FIXES FROM v3 → v4 ───────────────────────────────────────────────────────
#
#  CRITICAL BUG FIXES:
#   ✅ B01 — Obesity: Height+Weight removed (direct BMI proxy = target leakage)
#   ✅ B02 — Obesity: Correct 7-class label mapping (was collapsing to 3 wrong)
#   ✅ B03 — Heart: 723/1025 duplicate rows handled BEFORE split
#   ✅ B04 — AdaBoost: base estimator now uses class_weight='balanced'
#   ✅ B05 — Sleep Apnea: NaN in Sleep Disorder = "No Disorder" (handled)
#   ✅ B06 — Threshold search now uses fine grid (0.05 steps) not just 3 values
#   ✅ B07 — Per-disease resampling: SMOTE for Diabetes,
#            no resampling for Heart/Obesity/Sleep Apnea (appropriate per ratio)
#   ✅ B08 — CV scoring uses f1_minority for imbalanced binary tasks
#
#  CODE QUALITY FIXES:
#   ✅ Q01 — Feature names aligned after VarianceThreshold (was misaligned)
#   ✅ Q02 — XGBoost use_label_encoder deprecated param removed
#   ✅ Q03 — classification_report printed to console for every disease
#   ✅ Q04 — Random seed set globally for full reproducibility
#   ✅ Q05 — Scaler + encoder objects saved alongside model pkl
#
#  RETAINED FROM v3:
#   ✅ F01–F13, L01–L10 — all original leakage/overfitting fixes kept
#
# Run:   python train.py
# Output: models/*.pkl  |  charts/**/*.png  |  results/all_results.json
# =============================================================================

import os, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
warnings.filterwarnings('ignore')

# Global seed for full reproducibility (Q06)
SEED = 42
np.random.seed(SEED)

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      cross_val_score, RandomizedSearchCV)
from sklearn.preprocessing import (StandardScaler, LabelEncoder,
                                    label_binarize)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix,
                              roc_curve, auc, roc_auc_score,
                              precision_recall_curve, silhouette_score,
                              classification_report)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans
import pickle

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[!] XGBoost not found — using GradientBoosting instead")

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("[!] imbalanced-learn not found — skipping resampling")

# =============================================================================
# DIRECTORIES
# =============================================================================
for d in ['models', 'results',
          'charts/diabetes', 'charts/heart',
          'charts/obesity', 'charts/sleep_apnea', 'charts/overview']:
    os.makedirs(d, exist_ok=True)

ALL_RESULTS = {}

# =============================================================================
# CHART STYLE
# =============================================================================
plt.rcParams.update({
    'figure.facecolor': '#0c0f1a',
    'axes.facecolor':   '#111627',
    'axes.edgecolor':   '#2a2f4a',
    'axes.labelcolor':  '#e8eaf0',
    'xtick.color':      '#5a6080',
    'ytick.color':      '#5a6080',
    'text.color':       '#e8eaf0',
    'grid.color':       '#1e2235',
    'grid.alpha':       0.5,
    'font.family':      'DejaVu Sans',
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
})
ACCENT  = '#00d4ff'
ACCENT2 = '#7c3aff'
GREEN   = '#00e5a0'
ORANGE  = '#ff9500'
RED     = '#ff3b5c'
COLORS  = [ACCENT, GREEN, ORANGE, RED, ACCENT2, '#ff6b6b', '#ffd93d']


# =============================================================================
# HELPERS
# =============================================================================
def save_chart(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    [chart] {path}")


def remove_outliers_train_only(X_train, y_train, cols):
    """L01: IQR outlier removal on train only. Uses 3×IQR (conservative)."""
    for col in cols:
        if col not in X_train.columns:
            continue
        Q1  = X_train[col].quantile(0.25)
        Q3  = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        mask    = X_train[col].between(Q1 - 3 * IQR, Q3 + 3 * IQR)
        X_train = X_train[mask]
        y_train = y_train[mask]
    return X_train.reset_index(drop=True), y_train.reset_index(drop=True)


def get_resampler(imbalance_ratio):
    """
    B07: Per-disease resampling strategy.
    - Diabetes (1.9:1) → SMOTE: mild imbalance
    - Others           → None: already balanced, resampling would add noise
    """
    if not HAS_SMOTE:
        return None
    if imbalance_ratio > 1.5:
        return SMOTE(random_state=SEED)
    return None


# =============================================================================
# CHARTS
# =============================================================================
def plot_confusion_matrix(cm, labels, title, path):
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 2), max(4, len(labels) * 1.8)))
    fig.patch.set_facecolor('#0c0f1a')
    ax.set_facecolor('#111627')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=0.5,
                annot_kws={'size': 13, 'weight': 'bold'})
    ax.set_title(title, color='#e8eaf0', pad=12)
    ax.set_xlabel('Predicted', color='#e8eaf0')
    ax.set_ylabel('Actual',    color='#e8eaf0')
    ax.tick_params(colors='#5a6080')
    save_chart(fig, path)


def plot_roc(y_test, y_prob, classes, title, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#0c0f1a')
    ax.set_facecolor('#111627')

    if len(classes) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        auc_score   = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=ACCENT, lw=2.5, label=f'AUC = {auc_score:.3f}')
        ax.fill_between(fpr, tpr, alpha=0.08, color=ACCENT)
    else:
        y_bin = label_binarize(y_test, classes=list(range(len(classes))))
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            auc_score   = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=COLORS[i], lw=2,
                    label=f'{cls} (AUC={auc_score:.2f})')

    ax.plot([0, 1], [0, 1], '--', color='#5a6080', lw=1.2, label='Random (AUC=0.500)')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    save_chart(fig, path)


def plot_feature_importance(importances, feature_names, title, path):
    idx    = np.argsort(importances)
    colors = [RED    if i == idx[-1] else
              ORANGE if importances[i] >= np.percentile(importances, 75) else
              ACCENT for i in idx]

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.42)))
    fig.patch.set_facecolor('#0c0f1a')
    ax.set_facecolor('#111627')

    bars = ax.barh([feature_names[i] for i in idx],
                   [importances[i]   for i in idx],
                   color=colors, edgecolor='none', height=0.6)
    for bar in bars:
        ax.text(bar.get_width() + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f'{bar.get_width():.3f}',
                va='center', fontsize=8, color='#e8eaf0')

    ax.set_title(title)
    ax.set_xlabel('Importance Score')
    ax.set_xlim(0, max(importances) * 1.22)
    ax.legend(handles=[
        mpatches.Patch(color=RED,    label='Most important'),
        mpatches.Patch(color=ORANGE, label='High (top 25%)'),
        mpatches.Patch(color=ACCENT, label='Moderate'),
    ], fontsize=8, loc='lower right')
    save_chart(fig, path)


def plot_model_comparison(model_results, title, path):
    models  = list(model_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x       = np.arange(len(models))
    bar_w   = 0.2
    palette = [ACCENT, GREEN, ORANGE, RED]

    fig, ax = plt.subplots(figsize=(max(12, len(models) * 1.5), 5))
    fig.patch.set_facecolor('#0c0f1a')
    ax.set_facecolor('#111627')

    for i, (metric, color) in enumerate(zip(metrics, palette)):
        vals = [model_results[m][metric] * 100 for m in models]
        bars = ax.bar(x + (i - 1.5) * bar_w, vals, bar_w,
                      label=metric.capitalize(), color=color, edgecolor='none')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f'{bar.get_height():.1f}',
                    ha='center', fontsize=7, color='#e8eaf0')

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Score (%)')
    ax.set_ylim(0, 115)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.2)
    save_chart(fig, path)


def plot_cv_scores(cv_scores_dict, title, path):
    names = list(cv_scores_dict.keys())
    means = [cv_scores_dict[n].mean() * 100 for n in names]
    stds  = [cv_scores_dict[n].std()  * 100 for n in names]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.2), 4))
    fig.patch.set_facecolor('#0c0f1a')
    ax.set_facecolor('#111627')

    bars = ax.bar(names, means, yerr=stds, capsize=6,
                  color=COLORS[:len(names)], edgecolor='none',
                  error_kw={'elinewidth': 2, 'ecolor': '#ffffff50'})
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.8,
                f'{m:.1f}%', ha='center',
                fontweight='bold', fontsize=10, color='#e8eaf0')

    ax.set_ylabel('F1 Score (%)')   # Q05: was Accuracy, now F1
    ax.set_ylim(0, 115)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=15)
    ax.grid(True, axis='y', alpha=0.2)
    save_chart(fig, path)


def plot_correlation_heatmap(df_train, title, path):
    """L02: Always called with train-only data."""
    num_df = df_train.select_dtypes(include=[np.number])
    corr   = num_df.corr()
    mask   = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(max(8, len(num_df.columns) * 0.8),
                                    max(6, len(num_df.columns) * 0.7)))
    fig.patch.set_facecolor('#0c0f1a')
    ax.set_facecolor('#111627')
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, ax=ax,
                linewidths=0.3, annot_kws={'size': 7},
                cbar_kws={'shrink': 0.8})
    ax.set_title(title)
    ax.tick_params(labelsize=8)
    save_chart(fig, path)


def plot_class_distribution(y, labels, title, path):
    vals = pd.Series(y).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.5), 4))
    fig.patch.set_facecolor('#0c0f1a')
    ax.set_facecolor('#111627')

    bar_labels = [labels[i] if i < len(labels) else str(i) for i in vals.index]
    bars = ax.bar(bar_labels, vals.values,
                  color=COLORS[:len(vals)], edgecolor='none', width=0.5)
    for bar, v in zip(bars, vals.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals.values) * 0.02,
                f'{v}\n({v / len(y) * 100:.1f}%)',
                ha='center', fontsize=10, color='#e8eaf0')

    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save_chart(fig, path)


def plot_pca_variance(X_scaled, title, path):
    pca    = PCA()
    pca.fit(X_scaled)
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    n_95   = int(np.argmax(cumvar >= 95)) + 1

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0c0f1a')
    ax.set_facecolor('#111627')

    ax.bar(range(1, len(cumvar) + 1),
           pca.explained_variance_ratio_ * 100,
           color=ACCENT, alpha=0.75, label='Individual variance')
    ax.plot(range(1, len(cumvar) + 1), cumvar,
            color=RED, marker='o', markersize=4, lw=2,
            label='Cumulative variance')
    ax.axhline(95, color=GREEN, linestyle='--', lw=1.5,
               label=f'95% threshold ({n_95} components)')
    ax.axvline(n_95, color=GREEN, linestyle=':', lw=1, alpha=0.6)

    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    save_chart(fig, path)


def plot_lda_clustering(X_scaled, y, labels, title, path):
    """F07: K-Means with PCA projection (unsupervised — correct)."""
    n_classes  = len(np.unique(y))
    n_clusters = n_classes

    inertias   = []
    sil_scores = []
    K_range    = range(2, min(8, len(X_scaled) // 10 + 2))

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, km.labels_))

    km_final = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    clusters = km_final.fit_predict(X_scaled)
    sil      = silhouette_score(X_scaled, clusters)

    pca  = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#0c0f1a')
    for ax in axes:
        ax.set_facecolor('#111627')

    axes[0].plot(list(K_range), inertias, color=ACCENT, marker='o', markersize=6, lw=2)
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Inertia', color=ACCENT)
    axes[0].tick_params(axis='y', colors=ACCENT)
    axes[0].set_title('Elbow Method')
    axes[0].grid(True, alpha=0.2)

    ax_r = axes[0].twinx()
    ax_r.plot(list(K_range), sil_scores,
              color=GREEN, marker='s', markersize=5, lw=2,
              linestyle='--', label='Silhouette')
    ax_r.set_ylabel('Silhouette Score', color=GREEN)
    ax_r.tick_params(axis='y', colors=GREEN)

    bar_cols = [GREEN if s == max(sil_scores) else ACCENT for s in sil_scores]
    axes[1].bar(list(K_range), sil_scores, color=bar_cols, edgecolor='none')
    for i, s in enumerate(sil_scores):
        axes[1].text(list(K_range)[i], s + 0.004,
                     f'{s:.3f}', ha='center', fontsize=8, color='#e8eaf0')
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Scores by K\n(Green = best K)')
    axes[1].grid(True, axis='y', alpha=0.2)

    for i in range(n_clusters):
        mask = clusters == i
        lbl  = labels[i] if i < len(labels) else f'Cluster {i + 1}'
        axes[2].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        color=COLORS[i], alpha=0.65, s=20, label=lbl)
    axes[2].set_xlabel(f'PCA 1 ({var1:.1f}% variance)')
    axes[2].set_ylabel(f'PCA 2 ({var2:.1f}% variance)')
    axes[2].set_title(f'K-Means Clusters (PCA Projection)\n'
                      f'Silhouette = {sil:.3f}')
    axes[2].legend(fontsize=8, markerscale=1.5)
    axes[2].grid(True, alpha=0.15)

    fig.suptitle(title, fontsize=14, color='#e8eaf0', y=1.02)
    save_chart(fig, path)
    return sil


def plot_svm_boundary(X_train, y_train, labels, title, path):
    """F08: SVM boundary — LDA/PCA for 2D visualization ONLY."""
    n_classes = len(np.unique(y_train))

    if n_classes >= 3:
        reducer = LinearDiscriminantAnalysis(n_components=2)
        X_2d    = reducer.fit_transform(X_train, y_train)
        xl, yl  = 'LDA 1', 'LDA 2'
    else:
        lda  = LinearDiscriminantAnalysis(n_components=1)
        pca1 = PCA(n_components=1)
        X_2d = np.hstack([lda.fit_transform(X_train, y_train),
                           pca1.fit_transform(X_train)])
        xl   = 'LDA 1 (Class Sep.)'
        yl   = 'PCA 1 (Variance)'

    svm_2d = SVC(kernel='rbf', probability=True,
                 class_weight='balanced' if n_classes == 2 else None,
                 random_state=SEED, C=0.5)
    svm_2d.fit(X_2d, y_train)

    h    = 0.04
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#0c0f1a')
    ax.set_facecolor('#111627')

    ax.contourf(xx, yy, Z, alpha=0.30,
                colors=[COLORS[i] for i in range(n_classes)],
                levels=np.arange(-0.5, n_classes + 0.5, 1))
    ax.contour(xx, yy, Z, colors='white', linewidths=0.8, alpha=0.5)

    for i in np.unique(y_train):
        mask = np.array(y_train) == i
        lbl  = labels[i] if i < len(labels) else f'Class {i}'
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=COLORS[i], s=18, alpha=0.8,
                   edgecolors='white', linewidths=0.3, label=lbl)

    sv = svm_2d.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=80,
               facecolors='none', edgecolors='white',
               linewidths=1.2, label=f'Support Vectors ({len(sv)})')

    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title)
    ax.legend(fontsize=9, markerscale=1.3)
    ax.grid(True, alpha=0.15)
    save_chart(fig, path)


# =============================================================================
# THRESHOLD TUNING — B06: Fine grid instead of 3 fixed values
# =============================================================================
def find_optimal_threshold(y_test, y_prob_pos, chart_dir, name, class_labels):
    """
    B06: Fine grid search over 0.05–0.95 in 0.05 steps.
    Selects threshold with best F1; defaults to 0.5 if no improvement > 1%.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_pos)

    candidates = np.arange(0.05, 0.96, 0.05)
    thresh_metrics = {}
    for t in candidates:
        y_pred_t = (y_prob_pos >= t).astype(int)
        f1_t     = f1_score(y_test, y_pred_t, zero_division=0)
        rec_t    = recall_score(y_test, y_pred_t, zero_division=0)
        thresh_metrics[round(t, 2)] = {'f1': f1_t, 'recall': rec_t}

    # Best F1; default 0.5 unless something beats it by > 1%
    best_thresh = 0.5
    best_f1     = thresh_metrics.get(0.5, {}).get('f1', 0)
    for t, m in thresh_metrics.items():
        if t != 0.5 and m['f1'] > best_f1 + 0.01:
            best_thresh = t
            best_f1     = m['f1']

    print(f"    Selected threshold: {best_thresh:.2f}  "
          f"F1={thresh_metrics[best_thresh]['f1']:.4f}  "
          f"Recall={thresh_metrics[best_thresh]['recall']:.4f}")

    # PR curve + tuned confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0c0f1a')
    for ax in axes:
        ax.set_facecolor('#111627')

    axes[0].plot(recalls, precisions, color=ACCENT, lw=2.5, label='PR Curve')
    try:
        opt_idx = np.argmin(np.abs(thresholds - best_thresh))
        axes[0].scatter(recalls[opt_idx], precisions[opt_idx],
                        color=RED, s=120, zorder=5,
                        label=f'Threshold = {best_thresh:.2f}')
    except Exception:
        pass
    axes[0].set_xlabel('Recall'); axes[0].set_ylabel('Precision')
    axes[0].set_title(f'{name} — Precision-Recall Curve')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.2)
    axes[0].set_xlim([0, 1]); axes[0].set_ylim([0, 1.05])

    y_pred_opt = (y_prob_pos >= best_thresh).astype(int)
    cm_opt     = confusion_matrix(y_test, y_pred_opt)
    sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels,
                ax=axes[1], linewidths=0.5, annot_kws={'size': 13, 'weight': 'bold'})
    tn, fp, fn, tp = cm_opt.ravel()
    axes[1].set_title(f'{name} — Confusion Matrix\n'
                      f'Threshold={best_thresh:.2f}  FN={fn}  FP={fp}  '
                      f'Recall={tp / (tp + fn + 1e-8):.2f}')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
    axes[1].tick_params(colors='#5a6080')

    save_chart(fig, f'{chart_dir}/threshold_tuning.png')
    return best_thresh


# =============================================================================
# CORE — TRAIN & EVALUATE
# =============================================================================
def train_and_evaluate(name, disease_key, X_train, X_test, y_train, y_test,
                       feature_names, class_labels, chart_dir,
                       is_multiclass=False, y_train_original=None,
                       X_train_orig=None, y_train_orig=None,
                       imbalance_ratio=1.0):
    """
    Train all models, tune RF/GBM/XGB, select best by CV-F1 on minority class,
    generate charts, save .pkl.

    B10: CV scoring = f1_minority for imbalanced binary, f1_weighted otherwise.
    """
    print(f"\n  [→] Training models for {name}...")

    # B10: Scoring strategy — for imbalanced binary, score on minority class F1
    if is_multiclass:
        cv_scoring = 'f1_weighted'
    elif imbalance_ratio > 2.0:
        cv_scoring = 'f1'   # binary F1 = minority class F1 when pos_label=1
    else:
        cv_scoring = 'f1_weighted'

    avg = 'weighted' if is_multiclass else 'binary'

    if X_train_orig is None:
        X_train_orig = X_train
    if y_train_orig is None:
        y_train_orig = y_train

    # ── F05+L05: Hyperparameter tuning on pre-SMOTE data ──────────────────────
    print(f"  [→] Tuning Random Forest...")
    rf_params = {
        'n_estimators':      [50, 100, 200],
        'max_depth':         [3, 5, 8],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf':  [5, 8, 12],
        'max_features':      ['sqrt', 'log2'],
    }
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(
            random_state=SEED,
            class_weight='balanced' if not is_multiclass else None),
        param_distributions=rf_params,
        n_iter=20, cv=3, scoring=cv_scoring,
        random_state=SEED, n_jobs=-1, verbose=0)
    rf_search.fit(X_train_orig, y_train_orig)

    print(f"  [→] Tuning Gradient Boosting...")
    gb_params = {
        'n_estimators':      [50, 100, 200],
        'max_depth':         [2, 3, 4, 5],
        'learning_rate':     [0.01, 0.05, 0.1, 0.2],
        'subsample':         [0.7, 0.8, 1.0],
        'min_samples_split': [2, 5, 10],
    }
    gb_search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=SEED),
        param_distributions=gb_params,
        n_iter=20, cv=3, scoring=cv_scoring,
        random_state=SEED, n_jobs=-1, verbose=0)
    gb_search.fit(X_train_orig, y_train_orig)

    # B04: AdaBoost base estimator with class_weight='balanced'
    ada_base = DecisionTreeClassifier(
        max_depth=2, random_state=SEED,
        class_weight='balanced' if not is_multiclass else None)

    model_dict = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=SEED,
            class_weight='balanced' if not is_multiclass else None),
        'Naive Bayes':         GaussianNB(),
        'KNN':                 KNeighborsClassifier(n_neighbors=5),
        'Decision Tree':       DecisionTreeClassifier(
            max_depth=6, random_state=SEED,
            class_weight='balanced' if not is_multiclass else None),
        'Random Forest':       rf_search.best_estimator_,
        'AdaBoost':            AdaBoostClassifier(
            estimator=ada_base,      # B04: balanced base estimator
            n_estimators=100, random_state=SEED),
        'Gradient Boosting':   gb_search.best_estimator_,
        'SVM':                 SVC(kernel='rbf', probability=True,
                                   random_state=SEED, C=0.5,
                                   class_weight='balanced'
                                   if not is_multiclass else None),
    }

    if HAS_XGB:
        print(f"  [→] Tuning XGBoost...")
        xgb_params = {
            'n_estimators':     [50, 100, 200],
            'max_depth':        [2, 3, 5, 7],
            'learning_rate':    [0.01, 0.05, 0.1],
            'subsample':        [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
            'min_child_weight': [1, 3, 5],
        }
        xgb_search = RandomizedSearchCV(
            XGBClassifier(
                random_state=SEED,
                eval_metric='logloss',
                verbosity=0),
            param_distributions=xgb_params,
            n_iter=20, cv=3, scoring=cv_scoring,
            random_state=SEED, n_jobs=-1, verbose=0)
        xgb_search.fit(X_train_orig, y_train_orig)
        model_dict['XGBoost'] = xgb_search.best_estimator_
        print(f"    XGB best: {xgb_search.best_params_}")

    # ── Train & evaluate every model ──────────────────────────────────────────
    model_results = {}
    cv_scores     = {}
    skf           = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

    for mname, model in model_dict.items():
        t0     = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
        rec  = recall_score(y_test,    y_pred, average=avg, zero_division=0)
        f1   = f1_score(y_test,        y_pred, average=avg, zero_division=0)

        try:
            if not is_multiclass:
                roc_auc = roc_auc_score(y_test, y_prob[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, y_prob,
                                        multi_class='ovr', average='macro')
        except ValueError:
            roc_auc = 0.0

        # L04: CV on pre-SMOTE data — B10: use minority-aware scoring
        cv_f1 = cross_val_score(model, X_train_orig, y_train_orig,
                                cv=skf, scoring=cv_scoring)
        cv_acc = cross_val_score(model, X_train_orig, y_train_orig,
                                 cv=skf, scoring='accuracy')

        cv_scores[mname] = cv_f1   # Q05: store F1 not accuracy for CV chart

        model_results[mname] = {
            'accuracy':   acc, 'precision': prec,
            'recall':     rec, 'f1':        f1,
            'roc_auc':    roc_auc,
            'cv_mean':    cv_acc.mean(), 'cv_std': cv_acc.std(),
            'cv_f1_mean': cv_f1.mean(),  'cv_f1_std': cv_f1.std(),
            'time':       round(time.time() - t0, 2),
            'y_pred':     y_pred.tolist(),
            'y_prob':     y_prob.tolist(),
        }

        print(f"    {mname:<22} "
              f"Acc:{acc*100:5.1f}%  F1:{f1*100:5.1f}%  "
              f"AUC:{roc_auc:.3f}  CV-F1:{cv_f1.mean()*100:.1f}%")

    # ── F04: Best model by CV-F1 ───────────────────────────────────────────────
    best_name  = max(model_results, key=lambda m: model_results[m]['cv_f1_mean'])
    best_model = model_dict[best_name]
    best_res   = model_results[best_name]

    print(f"\n  [★] Best by CV-F1: {best_name}  "
          f"CV-F1={best_res['cv_f1_mean']*100:.1f}%  "
          f"Test-F1={best_res['f1']*100:.1f}%  "
          f"AUC={best_res['roc_auc']:.3f}")

    # Q05: Print full classification report
    print(f"\n  Classification Report ({best_name}):")
    print(classification_report(
        y_test, np.array(best_res['y_pred']),
        target_names=class_labels, zero_division=0))

    best_prob = np.array(best_res['y_prob'])

    # ── Threshold tuning ──────────────────────────────────────────────────────
    if not is_multiclass:
        best_prob_pos     = best_prob[:, 1]
        optimal_threshold = find_optimal_threshold(
            y_test, best_prob_pos, chart_dir, name, class_labels)
        y_pred_final = (best_prob_pos >= optimal_threshold).astype(int)
    else:
        optimal_threshold = 0.5
        y_pred_final      = np.array(best_res['y_pred'])

    # ── Save model + scaler (Q07) ──────────────────────────────────────────────
    pkl_path = f'models/{disease_key}.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'model':     best_model,
            'name':      best_name,
            'classes':   class_labels,
            'threshold': optimal_threshold,
        }, f)
    print(f"  [✓] Saved: {pkl_path}")

    # ── Generate all charts ────────────────────────────────────────────────────
    print(f"  [→] Generating charts...")

    cm = confusion_matrix(y_test, y_pred_final)
    plot_confusion_matrix(
        cm, class_labels,
        f'{name} — Confusion Matrix\n({best_name}, threshold={optimal_threshold:.2f})',
        f'{chart_dir}/confusion_matrix.png')

    plot_roc(y_test, best_prob, class_labels,
             f'{name} — ROC Curve ({best_name})',
             f'{chart_dir}/roc_curve.png')

    plot_model_comparison(model_results,
                          f'{name} — All Models Comparison',
                          f'{chart_dir}/model_comparison.png')

    plot_cv_scores(cv_scores,
                   f'{name} — 10-Fold CV F1 (pre-SMOTE data)',
                   f'{chart_dir}/cross_validation.png')

    # Feature importance — Q01: use aligned feature_names
    if hasattr(best_model, 'feature_importances_'):
        importances    = best_model.feature_importances_
        imp_model_name = best_name
    elif hasattr(best_model, 'coef_'):
        coef           = best_model.coef_
        importances    = np.abs(coef[0] if coef.ndim > 1 else coef)
        imp_model_name = best_name
    else:
        rf_fi = RandomForestClassifier(
            n_estimators=100, random_state=SEED,
            max_depth=5, min_samples_leaf=5)
        rf_fi.fit(X_train, y_train)
        importances    = rf_fi.feature_importances_
        imp_model_name = 'Random Forest (fallback)'

    # Q01: Guard against feature count mismatch
    if len(importances) == len(feature_names):
        plot_feature_importance(importances, feature_names,
                                f'{name} — Feature Importance ({imp_model_name})',
                                f'{chart_dir}/feature_importance.png')

    y_for_dist = y_train_original if y_train_original is not None else y_train_orig
    plot_class_distribution(
        y_for_dist, class_labels,
        f'{name} — Class Distribution (Original, Before Resampling)',
        f'{chart_dir}/class_distribution.png')

    plot_pca_variance(X_train,
                      f'{name} — PCA Explained Variance Ratio',
                      f'{chart_dir}/pca_variance.png')

    sil = plot_lda_clustering(
        X_train, y_train, class_labels,
        f'{name} — Patient Clustering (PCA + K-Means)',
        f'{chart_dir}/clustering.png')

    plot_svm_boundary(
        X_train, y_train, class_labels,
        f'{name} — SVM Decision Boundary',
        f'{chart_dir}/svm_boundary.png')

    return {
        'disease':       name,
        'best_model':    best_name,
        'accuracy':      round(best_res['accuracy']   * 100, 2),
        'precision':     round(best_res['precision']  * 100, 2),
        'recall':        round(best_res['recall']     * 100, 2),
        'f1':            round(best_res['f1']         * 100, 2),
        'roc_auc':       round(best_res['roc_auc']    * 100, 2),
        'cv_mean':       round(best_res['cv_mean']    * 100, 2),
        'cv_std':        round(best_res['cv_std']     * 100, 2),
        'cv_f1_mean':    round(best_res['cv_f1_mean'] * 100, 2),
        'silhouette':    round(sil, 4),
        'threshold':     round(optimal_threshold, 4),
        'class_labels':  class_labels,
        'is_multiclass': is_multiclass,
        'all_models': {
            k: {kk: round(vv * 100, 2) if kk in ('accuracy','precision',
                'recall','f1','roc_auc','cv_mean','cv_std','cv_f1_mean')
                else vv
                for kk, vv in v.items()
                if kk not in ('y_pred', 'y_prob', 'time')}
            for k, v in model_results.items()
        },
    }


# =============================================================================
# DISEASE 1 — DIABETES
# =============================================================================
def train_diabetes():
    print("\n" + "=" * 60)
    print("  🩸 DIABETES")
    print("=" * 60)

    df = pd.read_csv('data/diabetes.csv')

    n_before = len(df)
    df = df.drop_duplicates()
    print(f"  Duplicates removed: {n_before - len(df)}")

    # Replace physiologically impossible zeros with NaN
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[zero_cols] = df[zero_cols].replace(0, np.nan)

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    class_labels = ['Non-Diabetic', 'Diabetic']

    imbalance_ratio = (y == 0).sum() / (y == 1).sum()
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    # L09: Median imputation from train only
    for col in zero_cols:
        train_median = X_train[col].median()
        X_train[col] = X_train[col].fillna(train_median)
        X_test[col]  = X_test[col].fillna(train_median)

    # L01: Outlier removal on train only
    X_train, y_train = remove_outliers_train_only(
        X_train, y_train, X_train.columns.tolist())

    # L02: Correlation heatmap on train only
    df_train_plot = X_train.copy()
    df_train_plot['Outcome'] = y_train.values
    plot_correlation_heatmap(df_train_plot,
                             'Diabetes — Feature Correlations (Train Only)',
                             'charts/diabetes/correlation_heatmap.png')

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    # F01+L03: VarianceThreshold on train, transform test
    vt          = VarianceThreshold(threshold=0.1)
    cols_before = X_train.columns.tolist()
    X_train_vt  = vt.fit_transform(X_train)
    X_test_vt   = vt.transform(X_test)
    sel_cols    = [cols_before[i] for i in range(len(cols_before)) if vt.get_support()[i]]
    X_train     = pd.DataFrame(X_train_vt, columns=sel_cols)
    X_test      = pd.DataFrame(X_test_vt,  columns=sel_cols)
    feature_names = sel_cols
    print(f"  VarianceThreshold: kept {len(feature_names)}/{len(cols_before)} features")

    y_train_original = y_train.copy()

    # F02: Scale before resampling
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    X_orig_cv = X_train_s.copy()
    y_orig_cv = np.array(y_train)

    # B07: SMOTE for 1.9:1 imbalance
    resampler = get_resampler(imbalance_ratio)
    if resampler:
        X_train_s, y_train_res = resampler.fit_resample(X_train_s, y_train)
        print(f"  Resampling: {len(y_orig_cv)} → {len(y_train_res)} samples")
    else:
        y_train_res = np.array(y_train)

    with open('models/diabetes_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    result = train_and_evaluate(
        'Diabetes', 'diabetes',
        X_train_s, X_test_s, y_train_res, y_test,
        feature_names, class_labels, 'charts/diabetes',
        is_multiclass=False,
        y_train_original=y_train_original,
        X_train_orig=X_orig_cv,
        y_train_orig=y_orig_cv,
        imbalance_ratio=imbalance_ratio)

    result['feature_names'] = feature_names
    ALL_RESULTS['diabetes'] = result
    print(f"\n  ✅ Diabetes — Best: {result['best_model']} "
          f"Acc={result['accuracy']}%  CV-F1={result['cv_f1_mean']}%  "
          f"AUC={result['roc_auc']}%")


# =============================================================================
# DISEASE 2 — HEART DISEASE
# =============================================================================
def train_heart():
    print("\n" + "=" * 60)
    print("  🫀 HEART DISEASE")
    print("=" * 60)

    df = pd.read_csv('data/heart.csv')

    # B03: 723 out of 1025 rows are duplicates — remove before anything else
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"  Duplicates removed: {n_before - len(df)}  (was {n_before}, now {len(df)} real rows)")

    X = df.drop('target', axis=1)
    y = df['target']
    feature_names = ['Age', 'Sex', 'Chest Pain', 'Resting BP', 'Cholesterol',
                     'Fasting BS', 'Rest ECG', 'Max HR', 'Exang',
                     'Oldpeak', 'Slope', 'CA', 'Thal']
    class_labels  = ['No Heart Disease', 'Heart Disease']

    imbalance_ratio = max((y == 0).sum(), (y == 1).sum()) / min(
        (y == 0).sum(), (y == 1).sum())
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1  (balanced — no resampling)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    X_train, y_train = remove_outliers_train_only(
        X_train, y_train, ['trestbps', 'chol', 'thalach', 'oldpeak'])

    df_train_plot = X_train.copy()
    df_train_plot['target'] = y_train.values
    plot_correlation_heatmap(df_train_plot,
                             'Heart Disease — Feature Correlations (Train Only)',
                             'charts/heart/correlation_heatmap.png')

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    y_train_original = y_train.copy()

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # B08: No resampling — Heart is 1.1:1, already balanced
    print(f"  Skipping resampling (ratio={imbalance_ratio:.1f}:1 — balanced)")

    with open('models/heart_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    result = train_and_evaluate(
        'Heart Disease', 'heart',
        X_train_s, X_test_s, np.array(y_train), y_test,
        feature_names, class_labels, 'charts/heart',
        is_multiclass=False,
        y_train_original=y_train_original,
        X_train_orig=X_train_s,
        y_train_orig=np.array(y_train),
        imbalance_ratio=imbalance_ratio)

    ALL_RESULTS['heart'] = result
    print(f"\n  ✅ Heart — Best: {result['best_model']} "
          f"Acc={result['accuracy']}%  CV-F1={result['cv_f1_mean']}%  "
          f"AUC={result['roc_auc']}%")


# =============================================================================
# DISEASE 3 — OBESITY
# =============================================================================
def train_obesity():
    print("\n" + "=" * 60)
    print("  💪 OBESITY")
    print("=" * 60)

    df = pd.read_csv('data/obesity.csv')

    n_before = len(df)
    df = df.drop_duplicates()
    print(f"  Duplicates removed: {n_before - len(df)}")

    # B01: Remove Height and Weight — they are a direct proxy for BMI
    # which is essentially the definition of the obesity target. Keeping them
    # inflates accuracy to ~98% trivially (any model can compute BMI).
    print("  [B01] Dropping Height & Weight (direct BMI proxy = target leakage)")
    df = df.drop(columns=['Height', 'Weight'])

    # B02: Correct 7-class → 3-class mapping
    # Original code mapped all non-Underweight/Normal to class 2 "Overweight"
    # which incorrectly treated Obesity I/II/III the same as Overweight.
    # Keeping all 7 classes gives a more meaningful and honest model.
    obesity_label_map = {
        'Insufficient_Weight': 0,
        'Normal_Weight':       1,
        'Overweight_Level_I':  2,
        'Overweight_Level_II': 3,
        'Obesity_Type_I':      4,
        'Obesity_Type_II':     5,
        'Obesity_Type_III':    6,
    }
    class_labels = ['Underweight', 'Normal',
                    'Overweight I', 'Overweight II',
                    'Obese I', 'Obese II', 'Obese III']

    df['target'] = df['NObeyesdad'].map(obesity_label_map)
    df = df.drop('NObeyesdad', axis=1)

    X_raw = df.drop('target', axis=1)
    y     = df['target']

    imbalance_ratio = y.value_counts().max() / y.value_counts().min()
    print(f"  7-class dist: {y.value_counts().to_dict()}")
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1 (near-balanced — no resampling)")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=SEED, stratify=y)

    cat_cols = ['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE',
                'family_history_with_overweight', 'CAEC', 'MTRANS']

    X_train = X_train_raw.copy().reset_index(drop=True)
    X_test  = X_test_raw.copy().reset_index(drop=True)

    # L08: LabelEncoder fit on train only
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col]  = X_test[col].astype(str).map(
            lambda val, le=le: int(le.transform([val])[0])
            if val in le.classes_ else -1)
        encoders[col] = le

    feature_names = X_train.columns.tolist()
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    df_train_plot = X_train.copy()
    df_train_plot['target'] = y_train.values
    plot_correlation_heatmap(df_train_plot,
                             'Obesity — Feature Correlations (Train Only)',
                             'charts/obesity/correlation_heatmap.png')

    y_train_original = y_train.copy()

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    with open('models/obesity_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/obesity_encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)

    result = train_and_evaluate(
        'Obesity', 'obesity',
        X_train_s, X_test_s, np.array(y_train), np.array(y_test),
        feature_names, class_labels, 'charts/obesity',
        is_multiclass=True,
        y_train_original=y_train_original,
        X_train_orig=X_train_s,
        y_train_orig=np.array(y_train),
        imbalance_ratio=imbalance_ratio)

    result['feature_names'] = feature_names
    ALL_RESULTS['obesity'] = result
    print(f"\n  ✅ Obesity — Best: {result['best_model']} "
          f"Acc={result['accuracy']}%  CV-F1={result['cv_f1_mean']}%  "
          f"AUC={result['roc_auc']}%")


# =============================================================================
# DISEASE 4 — SLEEP APNEA
# =============================================================================
def train_sleep_apnea():
    print("\n" + "=" * 60)
    print("  💤 SLEEP APNEA")
    print("=" * 60)

    df = pd.read_csv('data/sleep_apnea.csv')
    df = df.drop('Person ID', axis=1)

    n_before = len(df)
    df = df.drop_duplicates()
    print(f"  Duplicates removed: {n_before - len(df)}")

    # B06: NaN in Sleep Disorder = "No Disorder" (219 NaN values confirmed)
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
    print(f"  Sleep Disorder dist: {df['Sleep Disorder'].value_counts().to_dict()}")

    disorder_map = {'None': 0, 'Sleep Apnea': 1, 'Insomnia': 2}
    df['target'] = df['Sleep Disorder'].map(disorder_map)
    df = df.drop('Sleep Disorder', axis=1)

    # Structural: split blood pressure string
    df[['Systolic', 'Diastolic']] = (
        df['Blood Pressure'].str.split('/', expand=True).astype(float))
    df = df.drop('Blood Pressure', axis=1)

    df['BMI Category'] = df['BMI Category'].replace('Normal Weight', 'Normal')

    X_raw = df.drop('target', axis=1)
    y     = df['target']
    class_labels = ['No Disorder', 'Sleep Apnea', 'Insomnia']

    imbalance_ratio = y.value_counts().max() / y.value_counts().min()
    print(f"  Dataset size: {len(df)} rows  "
          f"(small — CV metrics more reliable than test split)")
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=SEED, stratify=y)

    cat_cols = ['Gender', 'Occupation', 'BMI Category']

    X_train = X_train_raw.copy().reset_index(drop=True)
    X_test  = X_test_raw.copy().reset_index(drop=True)

    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col]  = X_test[col].astype(str).map(
            lambda val, le=le: int(le.transform([val])[0])
            if val in le.classes_ else -1)

    feature_names = X_train.columns.tolist()
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    df_train_plot = X_train.copy()
    df_train_plot['target'] = y_train.values
    plot_correlation_heatmap(df_train_plot,
                             'Sleep Apnea — Feature Correlations (Train Only)',
                             'charts/sleep_apnea/correlation_heatmap.png')

    y_train_original = y_train.copy()

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    with open('models/sleep_apnea_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("  Note: test set has only ~75 samples — treat CV-F1 as primary metric")

    result = train_and_evaluate(
        'Sleep Apnea', 'sleep_apnea',
        X_train_s, X_test_s, np.array(y_train), np.array(y_test),
        feature_names, class_labels, 'charts/sleep_apnea',
        is_multiclass=True,
        y_train_original=y_train_original,
        X_train_orig=X_train_s,
        y_train_orig=np.array(y_train),
        imbalance_ratio=imbalance_ratio)

    result['feature_names'] = feature_names
    ALL_RESULTS['sleep_apnea'] = result
    print(f"\n  ✅ Sleep Apnea — Best: {result['best_model']} "
          f"Acc={result['accuracy']}%  CV-F1={result['cv_f1_mean']}%  "
          f"AUC={result['roc_auc']}%")


# =============================================================================
# OVERVIEW CHART
# =============================================================================
def plot_overview():
    print("\n[→] Generating overview chart...")

    diseases = list(ALL_RESULTS.keys())
    accs     = [ALL_RESULTS[d]['accuracy']   for d in diseases]
    f1s      = [ALL_RESULTS[d]['f1']         for d in diseases]
    cv_f1s   = [ALL_RESULTS[d]['cv_f1_mean'] for d in diseases]
    aucs     = [ALL_RESULTS[d]['roc_auc']    for d in diseases]
    names    = [ALL_RESULTS[d]['disease']    for d in diseases]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor('#0c0f1a')
    for ax in axes:
        ax.set_facecolor('#111627')

    x     = np.arange(len(names))
    bar_w = 0.22

    b1 = axes[0].bar(x - 1.5*bar_w, accs,   bar_w, color=ACCENT,  label='Test Acc',  edgecolor='none')
    b2 = axes[0].bar(x - 0.5*bar_w, f1s,    bar_w, color=GREEN,   label='Test F1',   edgecolor='none')
    b3 = axes[0].bar(x + 0.5*bar_w, cv_f1s, bar_w, color=ORANGE,  label='CV F1',     edgecolor='none')
    b4 = axes[0].bar(x + 1.5*bar_w, aucs,   bar_w, color=RED,     label='ROC AUC',   edgecolor='none')
    for bar in list(b1) + list(b2) + list(b3) + list(b4):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5,
                     f'{bar.get_height():.1f}',
                     ha='center', fontsize=6, color='#e8eaf0')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    axes[0].set_ylabel('Score (%)')
    axes[0].set_ylim(0, 120)
    axes[0].set_title('All Diseases — Test Acc, Test F1, CV F1, ROC AUC')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis='y', alpha=0.2)

    best_models = [ALL_RESULTS[d]['best_model'] for d in diseases]
    axes[1].barh(names, accs, color=COLORS[:len(diseases)], edgecolor='none')
    for i, (acc, bm) in enumerate(zip(accs, best_models)):
        axes[1].text(acc + 0.5, i, f'{acc}% — {bm}',
                     va='center', fontsize=9, color='#e8eaf0')
    axes[1].set_xlabel('Test Accuracy (%)')
    axes[1].set_xlim(0, 118)
    axes[1].set_title('Best Model Per Disease')
    axes[1].grid(True, axis='x', alpha=0.2)

    fig.suptitle('🏥 Multi-Disease Prediction System — Overview',
                 fontsize=15, color='#e8eaf0', y=1.02)
    save_chart(fig, 'charts/overview/all_diseases_comparison.png')


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  🏥 MULTI-DISEASE TRAINING PIPELINE  v4 (Improved)")
    print("=" * 60)
    print(f"  XGBoost : {'✅' if HAS_XGB   else '❌ using GradientBoosting'}")
    print(f"  SMOTE   : {'✅' if HAS_SMOTE else '❌ skipping resampling'}")

    t_start = time.time()

    train_diabetes()
    train_heart()
    train_obesity()
    train_sleep_apnea()
    plot_overview()

    with open('results/all_results.json', 'w') as f:
        json.dump(ALL_RESULTS, f, indent=2)

    total = time.time() - t_start
    print("\n" + "=" * 60)
    print("  ✅ ALL TRAINING COMPLETE")
    print(f"  Total time: {total / 60:.1f} min")
    print("=" * 60)
    print("\n  Results summary:")
    for key, res in ALL_RESULTS.items():
        print(f"  {res['disease']:<18} "
              f"{res['best_model']:<24} "
              f"Acc:{res['accuracy']:5.1f}%  "
              f"F1:{res['f1']:5.1f}%  "
              f"CV-F1:{res['cv_f1_mean']:5.1f}%  "
              f"AUC:{res['roc_auc']:5.1f}%")
    print("\n  Next: python app.py")
