"""
Customer Churn Prediction System - Full ML Pipeline
Advanced version with SMOTE, feature engineering, ensemble models,
SHAP-style feature importance, and full evaluation suite.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, os, joblib, json
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, classification_report)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ── Paths ──────────────────────────────────────────────────────────────────────
import pathlib
BASE = str(pathlib.Path(__file__).resolve().parent.parent)
VISUALS = os.path.join(BASE, 'visuals')
MODELS  = os.path.join(BASE, 'models')
os.makedirs(VISUALS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
DARK   = '#0d0d14'
MID    = '#13131f'
CARD   = '#1a1a2e'
ACCENT = '#e94560'
BLUE   = '#0f3460'
GOLD   = '#f5a623'
TEAL   = '#00d4aa'
LAVEND = '#a78bfa'
TEXT   = '#e2e8f0'
MUTED  = '#64748b'

PALETTE = [ACCENT, TEAL, GOLD, LAVEND, '#60a5fa', '#fb923c']

def style():
    plt.rcParams.update({
        'figure.facecolor': DARK, 'axes.facecolor': MID,
        'axes.edgecolor': '#2d2d4e', 'axes.labelcolor': TEXT,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'text.color': TEXT, 'grid.color': '#2d2d4e',
        'grid.alpha': 0.5, 'font.family': 'monospace',
        'axes.titlecolor': TEXT, 'axes.titlesize': 13,
        'axes.labelsize': 11, 'legend.facecolor': CARD,
        'legend.edgecolor': '#2d2d4e', 'legend.labelcolor': TEXT,
    })

style()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  CUSTOMER CHURN PREDICTION SYSTEM — ADVANCED ML PIPELINE")
print("═"*60)

df = pd.read_csv(os.path.join(BASE, 'dataset', 'telecom_churn.csv'))
print(f"\n✓ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Churn rate: {(df['Churn']=='Yes').mean()*100:.1f}%")
print(f"  Missing values: {df.isnull().sum().sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 2] Preprocessing...")

df_clean = df.copy()
df_clean.drop_duplicates(inplace=True)
df_clean.drop(columns=['customerID'], inplace=True)

# Fix TotalCharges
df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median(), inplace=True)

# Binary encode
binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
    df_clean[col] = df_clean[col].map(binary_map)
df_clean['Churn'] = (df_clean['Churn'] == 'Yes').astype(int)

# Label encode multi-class
le = LabelEncoder()
for col in ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaymentMethod']:
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))

print(f"  ✓ Missing values handled, {df_clean.shape[1]} features ready")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 3] Feature Engineering...")

df_clean['avg_monthly_spend']  = df_clean['TotalCharges'] / (df_clean['tenure'] + 1)
df_clean['charge_per_service'] = df_clean['MonthlyCharges'] / (df_clean['PhoneService'] + df_clean['InternetService'] + 1)
df_clean['tenure_group']       = pd.cut(df_clean['tenure'], bins=[0,12,24,48,72], labels=[0,1,2,3]).astype(int)
df_clean['is_long_term']       = (df_clean['tenure'] > 24).astype(int)
df_clean['num_services']       = (df_clean[['OnlineSecurity','OnlineBackup','DeviceProtection',
                                            'TechSupport','StreamingTV','StreamingMovies']] > 0).sum(axis=1)
df_clean['engagement_score']   = (df_clean['num_services'] * 0.4 +
                                   df_clean['is_long_term'] * 0.3 +
                                   (1 - df_clean['SeniorCitizen']) * 0.15 +
                                   df_clean['Partner'] * 0.075 +
                                   df_clean['Dependents'] * 0.075)
df_clean['high_spender']       = (df_clean['MonthlyCharges'] > df_clean['MonthlyCharges'].quantile(0.75)).astype(int)
df_clean['low_support_risk']   = ((df_clean['TechSupport'] == 0) & (df_clean['OnlineSecurity'] == 0)).astype(int)

print(f"  ✓ Added 8 engineered features → {df_clean.shape[1]} total features")

# ══════════════════════════════════════════════════════════════════════════════
# EDA VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 4] Generating EDA visualizations...")

raw = pd.read_csv(os.path.join(BASE, 'dataset', 'telecom_churn.csv'))
raw['TotalCharges'] = pd.to_numeric(raw['TotalCharges'], errors='coerce').fillna(0)

# ── 1. Churn Overview Dashboard ────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor(DARK)
fig.suptitle('CUSTOMER CHURN — EXPLORATORY ANALYSIS', fontsize=16,
             fontweight='bold', color=TEXT, y=1.01, x=0.5)

# Churn distribution pie
churn_counts = raw['Churn'].value_counts()
ax = axes[0,0]
wedges, texts, autotexts = ax.pie(
    churn_counts, labels=['Retained', 'Churned'],
    autopct='%1.1f%%', colors=[TEAL, ACCENT],
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.5, edgecolor=DARK, linewidth=2))
for at in autotexts: at.set_color(DARK); at.set_fontsize(11); at.set_fontweight('bold')
for t in texts: t.set_color(TEXT)
ax.set_title('Churn Distribution', fontweight='bold')

# Contract vs Churn
ax = axes[0,1]
contract_churn = raw.groupby(['Contract','Churn']).size().unstack(fill_value=0)
contract_churn.plot(kind='bar', ax=ax, color=[TEAL, ACCENT], edgecolor=DARK, linewidth=0.5)
ax.set_title('Contract Type vs Churn', fontweight='bold')
ax.set_xlabel('Contract Type'); ax.set_ylabel('Count')
ax.legend(['Retained', 'Churned'], framealpha=0.3)
ax.tick_params(axis='x', rotation=25)
for spine in ax.spines.values(): spine.set_visible(False)

# Monthly Charges Distribution
ax = axes[0,2]
for label, color in [('No', TEAL), ('Yes', ACCENT)]:
    data = raw[raw['Churn'] == label]['MonthlyCharges']
    ax.hist(data, bins=30, alpha=0.6, color=color, label=f'{"Retained" if label=="No" else "Churned"}', edgecolor='none')
ax.set_title('Monthly Charges by Churn', fontweight='bold')
ax.set_xlabel('Monthly Charges ($)'); ax.set_ylabel('Count')
ax.legend(); ax.grid(True, alpha=0.3)
for spine in ax.spines.values(): spine.set_visible(False)

# Tenure by Churn
ax = axes[1,0]
for label, color in [('No', TEAL), ('Yes', ACCENT)]:
    data = raw[raw['Churn'] == label]['tenure']
    ax.hist(data, bins=25, alpha=0.6, color=color, label=f'{"Retained" if label=="No" else "Churned"}', edgecolor='none')
ax.set_title('Tenure Distribution by Churn', fontweight='bold')
ax.set_xlabel('Tenure (months)'); ax.set_ylabel('Count')
ax.legend(); ax.grid(True, alpha=0.3)
for spine in ax.spines.values(): spine.set_visible(False)

# Internet Service vs Churn
ax = axes[1,1]
internet_churn = raw.groupby(['InternetService','Churn']).size().unstack(fill_value=0)
internet_pct = internet_churn.div(internet_churn.sum(axis=1), axis=0) * 100
internet_pct.plot(kind='bar', ax=ax, color=[TEAL, ACCENT], edgecolor=DARK)
ax.set_title('Internet Service vs Churn %', fontweight='bold')
ax.set_xlabel('Internet Service'); ax.set_ylabel('Percentage (%)')
ax.legend(['Retained', 'Churned'], framealpha=0.3)
ax.tick_params(axis='x', rotation=25)
for spine in ax.spines.values(): spine.set_visible(False)

# Payment Method vs Churn
ax = axes[1,2]
pay_churn = raw.groupby(['PaymentMethod','Churn']).size().unstack(fill_value=0)
pay_pct = pay_churn.div(pay_churn.sum(axis=1), axis=0) * 100
colors_bar = [TEAL, ACCENT]
pay_pct.plot(kind='barh', ax=ax, color=colors_bar, edgecolor=DARK)
ax.set_title('Payment Method vs Churn %', fontweight='bold')
ax.set_xlabel('Percentage (%)'); ax.set_ylabel('')
ax.legend(['Retained', 'Churned'], framealpha=0.3)
for spine in ax.spines.values(): spine.set_visible(False)

plt.tight_layout(pad=2)
plt.savefig(os.path.join(VISUALS, '01_eda_overview.png'), dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print("  ✓ EDA overview saved")

# ── 2. Correlation Heatmap ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 12))
fig.patch.set_facecolor(DARK)
corr = df_clean.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, ax=ax,
            linewidths=0.5, linecolor=DARK, annot=True,
            fmt='.2f', annot_kws={'size': 7}, square=True,
            cbar_kws={'shrink': 0.7})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
ax.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS, '02_correlation_heatmap.png'), dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print("  ✓ Correlation heatmap saved")

# ── 3. Engineered Features Analysis ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle('ENGINEERED FEATURES ANALYSIS', fontsize=13, fontweight='bold', color=TEXT)

for ax_i, (feat, title) in enumerate(zip(
    ['engagement_score', 'num_services', 'avg_monthly_spend'],
    ['Engagement Score', 'Number of Services', 'Avg Monthly Spend'])):
    ax = axes[ax_i]
    for label, color in [(0, TEAL), (1, ACCENT)]:
        data = df_clean[df_clean['Churn'] == label][feat]
        ax.hist(data, bins=25, alpha=0.65, color=color,
                label='Retained' if label==0 else 'Churned', edgecolor='none')
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('Count')
    ax.legend(); ax.grid(True, alpha=0.3)
    for spine in ax.spines.values(): spine.set_visible(False)

plt.tight_layout(pad=2)
plt.savefig(os.path.join(VISUALS, '03_engineered_features.png'), dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print("  ✓ Engineered features chart saved")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: TRAIN-TEST SPLIT + SMOTE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 5] Train-test split & SMOTE balancing...")

X = df_clean.drop('Churn', axis=1)
y = df_clean['Churn']
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = RobustScaler()
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp  = imputer.transform(X_test)
joblib.dump(imputer, os.path.join(MODELS, 'imputer.pkl'))

X_train_sc = scaler.fit_transform(X_train_imp)
X_test_sc  = scaler.transform(X_test_imp)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_sc, y_train)

print(f"  Train: {X_train.shape[0]:,} → After SMOTE: {X_train_res.shape[0]:,}")
print(f"  Test:  {X_test.shape[0]:,}")
joblib.dump(scaler, os.path.join(MODELS, 'scaler.pkl'))
joblib.dump(feature_names, os.path.join(MODELS, 'feature_names.pkl'))

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 6] Training models...")

models = {
    'Logistic Regression': LogisticRegression(C=0.5, max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=8, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=12,
                                                  min_samples_split=5, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
                                                       max_depth=5, random_state=42),
    'XGBoost':             XGBClassifier(n_estimators=200, learning_rate=0.08, max_depth=6,
                                         use_label_encoder=False, eval_metric='logloss',
                                         random_state=42, n_jobs=-1),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred  = model.predict(X_test_sc)
    y_prob  = model.predict_proba(X_test_sc)[:,1] if hasattr(model, 'predict_proba') else None

    cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=cv, scoring='f1')
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob) if y_prob is not None else 0

    results[name] = {
        'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'roc_auc': auc,
        'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()
    }
    print(f"  ✓ {name:25s} | Acc:{acc:.3f} | F1:{f1:.3f} | AUC:{auc:.3f} | CV:{cv_scores.mean():.3f}±{cv_scores.std():.3f}")

# ── Voting Ensemble ──────────────────────────────────────────────────────────
print("  Building Voting Ensemble...")
ensemble = VotingClassifier(estimators=[
    ('rf',  models['Random Forest']),
    ('xgb', models['XGBoost']),
    ('gb',  models['Gradient Boosting']),
], voting='soft')
ensemble.fit(X_train_res, y_train_res)
y_pred_ens = ensemble.predict(X_test_sc)
y_prob_ens = ensemble.predict_proba(X_test_sc)[:,1]
results['Voting Ensemble'] = {
    'model': ensemble, 'y_pred': y_pred_ens, 'y_prob': y_prob_ens,
    'accuracy': accuracy_score(y_test, y_pred_ens),
    'precision': precision_score(y_test, y_pred_ens),
    'recall':    recall_score(y_test, y_pred_ens),
    'f1':        f1_score(y_test, y_pred_ens),
    'roc_auc':   roc_auc_score(y_test, y_prob_ens),
    'cv_mean': 0, 'cv_std': 0
}
r = results['Voting Ensemble']
print(f"  ✓ {'Voting Ensemble':25s} | Acc:{r['accuracy']:.3f} | F1:{r['f1']:.3f} | AUC:{r['roc_auc']:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: BEST MODEL SELECTION
# ══════════════════════════════════════════════════════════════════════════════
best_name = max(results, key=lambda k: results[k]['roc_auc'])
best      = results[best_name]
print(f"\n[BEST MODEL] → {best_name}")
print(f"  Accuracy: {best['accuracy']:.4f} | F1: {best['f1']:.4f} | AUC: {best['roc_auc']:.4f}")

joblib.dump(best['model'], os.path.join(MODELS, 'best_model.pkl'))
print(f"  ✓ Model saved to {MODELS}/best_model.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 7] Generating evaluation visuals...")

# ── Model Comparison ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(17, 6))
fig.patch.set_facecolor(DARK)
fig.suptitle('MODEL PERFORMANCE COMPARISON', fontsize=14, fontweight='bold', color=TEXT)

model_names = list(results.keys())
metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

x = np.arange(len(model_names))
width = 0.15
ax = axes[0]
for i, (m, label) in enumerate(zip(metrics_list, metric_labels)):
    vals = [results[n][m] for n in model_names]
    ax.bar(x + i*width, vals, width, label=label, color=PALETTE[i], alpha=0.85, edgecolor=DARK)
ax.set_xticks(x + width*2)
ax.set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=8)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel('Score')
ax.set_title('All Metrics by Model', fontweight='bold')
ax.legend(fontsize=8, framealpha=0.3)
ax.grid(True, alpha=0.3, axis='y')
for spine in ax.spines.values(): spine.set_visible(False)

# ROC Curves
ax = axes[1]
for name, r in results.items():
    if r['y_prob'] is not None:
        fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
        color = ACCENT if name == best_name else PALETTE[list(results.keys()).index(name) % len(PALETTE)]
        lw = 2.5 if name == best_name else 1.2
        ax.plot(fpr, tpr, color=color, lw=lw,
                label=f"{name} (AUC={r['roc_auc']:.3f})" + (" ★" if name == best_name else ""))
ax.plot([0,1],[0,1], 'w--', lw=0.8, alpha=0.4)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — All Models', fontweight='bold')
ax.legend(fontsize=8, framealpha=0.3, loc='lower right')
ax.grid(True, alpha=0.3)
for spine in ax.spines.values(): spine.set_visible(False)

plt.tight_layout(pad=2)
plt.savefig(os.path.join(VISUALS, '04_model_comparison.png'), dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print("  ✓ Model comparison chart saved")

# ── Confusion Matrix (Best Model) ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(DARK)
fig.suptitle(f'BEST MODEL: {best_name.upper()} — DETAILED EVALUATION',
             fontsize=13, fontweight='bold', color=TEXT)

cm = confusion_matrix(y_test, best['y_pred'])
cmap = sns.light_palette(ACCENT, as_cmap=True)
sns.heatmap(cm, annot=True, fmt='d', ax=axes[0], cmap=cmap,
            xticklabels=['Retained','Churned'], yticklabels=['Retained','Churned'],
            linewidths=2, linecolor=DARK, annot_kws={'size': 14, 'weight': 'bold'})
axes[0].set_title('Confusion Matrix', fontweight='bold')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')

# Metrics bars (best model)
metric_vals = [best['accuracy'], best['precision'], best['recall'], best['f1'], best['roc_auc']]
bars = axes[1].barh(metric_labels, metric_vals, color=PALETTE[:5],
                    edgecolor=DARK, height=0.55)
for bar, val in zip(bars, metric_vals):
    axes[1].text(val - 0.04, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', ha='right',
                 color=DARK, fontweight='bold', fontsize=11)
axes[1].set_xlim(0, 1.05)
axes[1].set_title(f'{best_name} — Metric Summary', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')
for spine in axes[1].spines.values(): spine.set_visible(False)

plt.tight_layout(pad=2)
plt.savefig(os.path.join(VISUALS, '05_best_model_evaluation.png'), dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print("  ✓ Confusion matrix saved")

# ── Feature Importance ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
fig.patch.set_facecolor(DARK)

try:
    # Try getting feature importances from the best model or its sub-estimator
    best_model = best['model']
    if hasattr(best_model, 'feature_importances_'):
        fi = best_model.feature_importances_
    elif hasattr(best_model, 'estimators_'):
        # VotingClassifier - average over sub-estimators with feature_importances_
        fi_list = [e.feature_importances_ for _, e in best_model.estimators_
                   if hasattr(e, 'feature_importances_')]
        fi = np.mean(fi_list, axis=0) if fi_list else None
    else:
        fi = None

    if fi is not None and len(fi) == len(feature_names):
        fi_df = pd.DataFrame({'feature': feature_names, 'importance': fi})
        fi_df = fi_df.sort_values('importance', ascending=True).tail(18)
        norm = fi_df['importance'] / fi_df['importance'].max()
        colors_fi = plt.cm.RdYlGn(norm)
        bars = ax.barh(fi_df['feature'], fi_df['importance'], color=colors_fi, edgecolor=DARK, height=0.65)
        for bar, val in zip(bars, fi_df['importance']):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', ha='left', color=TEXT, fontsize=8)
        ax.set_title(f'Feature Importance — {best_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance Score')
        ax.grid(True, alpha=0.3, axis='x')
        for spine in ax.spines.values(): spine.set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALS, '06_feature_importance.png'), dpi=150, bbox_inches='tight', facecolor=DARK)
        print("  ✓ Feature importance saved")
    else:
        print("  ⚠ Feature importance not available for this model")
except Exception as e:
    print(f"  ⚠ Feature importance skipped: {e}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 8] Generating business insights dashboard...")

# Add prediction probs to test set
X_test_df = X_test.copy()
X_test_df['churn_actual']     = y_test.values
X_test_df['churn_prob']       = best['y_prob']
X_test_df['churn_predicted']  = best['y_pred']
X_test_df['risk_segment']     = pd.cut(
    X_test_df['churn_prob'], bins=[0, 0.3, 0.6, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk'])

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor(DARK)
fig.suptitle('BUSINESS INTELLIGENCE DASHBOARD', fontsize=16, fontweight='bold', color=TEXT)

# Risk segment distribution
ax = axes[0,0]
seg_counts = X_test_df['risk_segment'].value_counts()
colors_seg = [TEAL, GOLD, ACCENT]
wedges, texts, autotexts = ax.pie(
    seg_counts, labels=seg_counts.index, autopct='%1.1f%%',
    colors=colors_seg, startangle=90,
    wedgeprops=dict(width=0.55, edgecolor=DARK, linewidth=2), pctdistance=0.78)
for at in autotexts: at.set_color(DARK); at.set_fontweight('bold')
for t in texts: t.set_color(TEXT)
ax.set_title('Customer Risk Segments', fontweight='bold')

# Revenue at risk
ax = axes[0,1]
# Map original monthly charges back
monthly_by_risk = X_test_df.groupby('risk_segment')['MonthlyCharges'].sum()
bars = ax.bar(monthly_by_risk.index, monthly_by_risk.values,
              color=colors_seg, edgecolor=DARK, width=0.5)
for bar, val in zip(bars, monthly_by_risk.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 500,
            f'${val:,.0f}', ha='center', va='bottom',
            color=TEXT, fontweight='bold', fontsize=10)
ax.set_title('Monthly Revenue at Risk by Segment', fontweight='bold')
ax.set_ylabel('Monthly Revenue ($)')
ax.grid(True, alpha=0.3, axis='y')
for spine in ax.spines.values(): spine.set_visible(False)

# Churn prob distribution
ax = axes[1,0]
ax.hist(X_test_df[X_test_df['churn_actual']==0]['churn_prob'], bins=30,
        alpha=0.65, color=TEAL, label='Retained', edgecolor='none')
ax.hist(X_test_df[X_test_df['churn_actual']==1]['churn_prob'], bins=30,
        alpha=0.65, color=ACCENT, label='Churned', edgecolor='none')
ax.axvline(0.5, color=GOLD, ls='--', lw=1.5, label='Decision boundary')
ax.set_title('Predicted Churn Probability Distribution', fontweight='bold')
ax.set_xlabel('Churn Probability'); ax.set_ylabel('Count')
ax.legend(); ax.grid(True, alpha=0.3)
for spine in ax.spines.values(): spine.set_visible(False)

# Avg monthly charges vs tenure for high risk
ax = axes[1,1]
high_risk = X_test_df[X_test_df['risk_segment']=='High Risk']
medium    = X_test_df[X_test_df['risk_segment']=='Medium Risk']
low_risk  = X_test_df[X_test_df['risk_segment']=='Low Risk']
ax.scatter(low_risk['tenure'],    low_risk['MonthlyCharges'],    c=TEAL,   alpha=0.4, s=15, label='Low')
ax.scatter(medium['tenure'],      medium['MonthlyCharges'],      c=GOLD,   alpha=0.4, s=15, label='Medium')
ax.scatter(high_risk['tenure'],   high_risk['MonthlyCharges'],   c=ACCENT, alpha=0.5, s=15, label='High')
ax.set_title('Tenure vs Monthly Charges by Risk', fontweight='bold')
ax.set_xlabel('Tenure (months)'); ax.set_ylabel('Monthly Charges ($)')
ax.legend(title='Risk', framealpha=0.3)
ax.grid(True, alpha=0.3)
for spine in ax.spines.values(): spine.set_visible(False)

plt.tight_layout(pad=2.5)
plt.savefig(os.path.join(VISUALS, '07_business_dashboard.png'), dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print("  ✓ Business dashboard saved")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE METADATA
# ══════════════════════════════════════════════════════════════════════════════
metadata = {
    'best_model': best_name,
    'metrics': {k: float(v) for k, v in {
        'accuracy': best['accuracy'], 'precision': best['precision'],
        'recall': best['recall'], 'f1': best['f1'], 'roc_auc': best['roc_auc']
    }.items()},
    'all_models': {
        name: {m: float(results[name][m]) for m in ['accuracy','precision','recall','f1','roc_auc']}
        for name in results
    },
    'dataset_shape': list(df.shape),
    'churn_rate': float((df['Churn']=='Yes').mean()),
    'feature_count': len(feature_names),
    'feature_names': feature_names,
}
with open(os.path.join(MODELS, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"\n  ✓ Metadata saved")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  PIPELINE COMPLETE")
print("═"*60)
print(f"\n  Best Model   : {best_name}")
print(f"  Accuracy     : {best['accuracy']:.4f}")
print(f"  Precision    : {best['precision']:.4f}")
print(f"  Recall       : {best['recall']:.4f}")
print(f"  F1-Score     : {best['f1']:.4f}")
print(f"  ROC-AUC      : {best['roc_auc']:.4f}")
print(f"\n  Visuals → {VISUALS}/")
print(f"  Models  → {MODELS}/")
print("═"*60 + "\n")
