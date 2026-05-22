"""
Volume Prediction ML Pipeline — State of the Art  (v4)
=======================================================
Input features : Kiln_Major_Axis, Kiln_Minor_Axis, Biochar_Major_Axis,
                 Biochar_Minor_Axis, Mean_Distance, Scaling_Factor,
                 Calculated_Volume
Target         : Volume (mL)
Objectives     : RMSE ~25-30 | MAE ~25-30 | R² > 90 %

Collinearity strategy
─────────────────────
• Raw features — severe multicollinearity confirmed (VIF: Kiln_Major=73,
  SF=43, Biochar_Major=15, Biochar_Minor=10, Kiln_Minor=9).
• Physics-based engineering converts pixel→mm, eliminating the dominant
  Kiln_Major↔Scaling_Factor collinearity (r=0.984) and creating meaningful
  orthogonal predictors.
• Linear models (Ridge/Lasso/ElasticNet) use VIF < 5 iteratively-selected
  features.
• Tree / kernel models are VIF-immune (one-feature-per-split); all curated
  features are used, VIF is reported for transparency.

Validation     : Stratified 80/20 split + 10-Fold CV + exact LOOCV (Ridge)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import (KFold, LeaveOneOut, cross_val_score,
                                     learning_curve, train_test_split,
                                     cross_validate, RandomizedSearchCV)
from sklearn.linear_model import (Ridge, Lasso, ElasticNet, RidgeCV,
                                   LinearRegression)
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor, HistGradientBoostingRegressor,
                               StackingRegressor)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ──────────────────────────────────────────────────────────────
# 0. Configuration
# ──────────────────────────────────────────────────────────────
DATA_PATH    = '/Users/mipl/Downloads/manual_after_merging.csv'
OUTPUT_DIR   = '/Users/mipl/Documents/Biochar/Artisanal/Model_v4'
RANDOM_STATE = 42
TEST_SIZE    = 0.20
K_FOLDS      = 10

RAW_FEATURES = [
    'Kiln_Major_Axis', 'Kiln_Minor_Axis',
    'Biochar_Major_Axis', 'Biochar_Minor_Axis',
    'Mean_Distance', 'Scaling_Factor',
    'Calculated_Volume',
]
TARGET = 'Volume'
np.random.seed(RANDOM_STATE)

# ──────────────────────────────────────────────────────────────
# 1. Data Loading
# ──────────────────────────────────────────────────────────────
print("=" * 70)
print("VOLUME PREDICTION PIPELINE  v3")
print("=" * 70)

df = pd.read_csv(DATA_PATH, index_col=0)
df = df[RAW_FEATURES + [TARGET]].dropna().reset_index(drop=True)
y  = df[TARGET]

print(f"\n[DATA]  n={len(df)} | features={len(RAW_FEATURES)} | target={TARGET}")
print(f"        Volume: {df[TARGET].min():.0f}–{df[TARGET].max():.0f} mL "
      f"(mean={df[TARGET].mean():.1f}, std={df[TARGET].std():.1f})")
print(f"        Discrete bins (50 mL steps): "
      f"{sorted(df[TARGET].unique().astype(int))}")

cv = df['Calculated_Volume']
print(f"\n[INFO]  Calculated_Volume: range {cv.min():.1f}–{cv.max():.1f}, "
      f"r={cv.corr(y):.4f} with Volume (standalone RMSE≈106)")

print("\n[EDA]  Pearson r with Volume:")
for f in RAW_FEATURES:
    r, p = stats.pearsonr(df[f], y)
    print(f"       {f:<25s}  r = {r:+.4f}  (p={p:.1e})")

# ──────────────────────────────────────────────────────────────
# 2. VIF helpers
# ──────────────────────────────────────────────────────────────
def compute_vif(X_df):
    sc  = StandardScaler()
    Xs  = pd.DataFrame(sc.fit_transform(X_df), columns=X_df.columns)
    return pd.DataFrame({
        'Feature': X_df.columns,
        'VIF'    : [variance_inflation_factor(Xs.values, i) for i in range(Xs.shape[1])]
    }).sort_values('VIF', ascending=False).reset_index(drop=True)

def vif_selection(X_df, threshold=5.0):
    X, removed = X_df.copy(), []
    while True:
        vif   = compute_vif(X)
        worst = vif.iloc[0]
        if worst['VIF'] <= threshold:
            break
        removed.append(worst['Feature'])
        X = X.drop(columns=[worst['Feature']])
    return X, compute_vif(X), removed

# ──────────────────────────────────────────────────────────────
# 3. Collinearity Analysis — raw features
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("COLLINEARITY ANALYSIS — RAW FEATURES")
print("=" * 70)
vif_raw = compute_vif(df[RAW_FEATURES])
print(vif_raw.to_string(index=False))
print("\n  ⚠  Kiln_Major_Axis ↔ Scaling_Factor collinearity: r=0.984 (VIF>40)")
print("  ⚠  Biochar_Minor_Axis ↔ Kiln_Minor_Axis:          r=0.820 (VIF>9)")
print("  → Resolving via physics-based feature engineering (pixel → mm conversion)")

# ──────────────────────────────────────────────────────────────
# 4. Feature Engineering  (physics-based)
# ──────────────────────────────────────────────────────────────
def engineer_features(df_in):
    """
    Physics-motivated features derived solely from the 6 raw inputs.
    Converting pixel→real-world (mm) removes the dominant collinearity.
    """
    X  = df_in[RAW_FEATURES].copy()
    SF = X['Scaling_Factor']      # pixels per mm

    # ── Real-world dimensions (mm) ──────────────────────────
    X['Biochar_Major_mm'] = X['Biochar_Major_Axis'] / SF
    X['Biochar_Minor_mm'] = X['Biochar_Minor_Axis'] / SF
    X['Kiln_Major_mm']    = X['Kiln_Major_Axis']    / SF
    X['Kiln_Minor_mm']    = X['Kiln_Minor_Axis']    / SF
    X['Distance_mm']      = X['Mean_Distance']      / SF

    # ── Shape / aspect ratios (dimensionless) ───────────────
    X['Biochar_AR']       = X['Biochar_Major_mm'] / (X['Biochar_Minor_mm'] + 1e-9)
    X['Kiln_AR']          = X['Kiln_Major_mm']    / (X['Kiln_Minor_mm']    + 1e-9)

    # ── 2-D projected areas (mm²) ───────────────────────────
    X['Biochar_Area']     = (np.pi / 4) * X['Biochar_Major_mm'] * X['Biochar_Minor_mm']
    X['Kiln_Area']        = (np.pi / 4) * X['Kiln_Major_mm']    * X['Kiln_Minor_mm']

    # ── Fill fractions (SF cancels in pixel ratio) ──────────
    X['Fill_Major']       = X['Biochar_Major_Axis'] / (X['Kiln_Major_Axis'] + 1e-9)
    X['Fill_Minor']       = X['Biochar_Minor_Axis'] / (X['Kiln_Minor_Axis'] + 1e-9)
    X['Area_Ratio']       = X['Biochar_Area'] / (X['Kiln_Area'] + 1e-9)

    # ── Physics volume proxy  (prolate spheroid V = π/6·a·b²) ─
    X['Vol_spheroid']     = (np.pi / 6) * X['Biochar_Major_mm'] * X['Biochar_Minor_mm']**2

    # ── Normalised distance ──────────────────────────────────
    X['Distance_norm']    = X['Distance_mm'] / (X['Kiln_Major_mm'] + 1e-9)
    X['Distance_sq_mm']   = X['Distance_mm']**2

    # ── Log-transformed distance (wide range 4–168 px) ──────
    X['Log_Mean_Dist']    = np.log1p(X['Mean_Distance'])
    X['Log_Dist_mm']      = np.log1p(X['Distance_mm'])
    X['Log_Dist_norm']    = np.log1p(X['Distance_norm'])

    # ── Compound interactions ────────────────────────────────
    X['Vol_x_Dist']       = X['Vol_spheroid'] * X['Distance_norm']
    X['Area_x_Dist']      = X['Area_Ratio']   * X['Distance_norm']
    X['Fill_x_Dist']      = X['Fill_Major']   * X['Distance_norm']
    X['Fill_x_AR']        = X['Fill_Minor']   * X['Biochar_AR']
    X['Sqrt_Dist_norm']   = np.sqrt(X['Distance_norm'])

    # ── Calculated_Volume features ───────────────────────────
    # The formula value can be negative; shift to make log-safe
    cv_min = X['Calculated_Volume'].min()
    X['CalcVol_log']      = np.log(X['Calculated_Volume'] - cv_min + 1)
    X['CalcVol_sq']       = X['Calculated_Volume'] ** 2
    # Residual: how much does the formula differ from the spheroid estimate
    X['CalcVol_vs_sph']   = X['Calculated_Volume'] - X['Vol_spheroid']

    return X

Xeng = engineer_features(df)
print(f"\n[FEAT] Engineered feature count: {Xeng.shape[1]}")

# ──────────────────────────────────────────────────────────────
# 5a. Linear-model features  (VIF < 5, strict)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("VIF SELECTION FOR LINEAR MODELS  (threshold = 5)")
print("=" * 70)
X_lin, vif_lin, removed_lin = vif_selection(Xeng, threshold=5.0)
print(f"Retained {X_lin.shape[1]} features (all VIF < 5):")
print(vif_lin.to_string(index=False))

# ──────────────────────────────────────────────────────────────
# 5b. Curated feature set for tree/kernel models
#     Selected by Mutual Information ranking (top non-redundant)
# ──────────────────────────────────────────────────────────────
# First compute MI on all engineered features to guide selection
mi_all = mutual_info_regression(Xeng, y, random_state=RANDOM_STATE)
mi_df  = pd.DataFrame({'Feature': Xeng.columns, 'MI': mi_all}) \
            .sort_values('MI', ascending=False)
print("\n[MI]  Top-15 features by Mutual Information:")
print(mi_df.head(15).to_string(index=False))

# Hand-picked: best MI, physically meaningful, minimal redundancy
TREE_FEATURES = [
    'Calculated_Volume',  # formula-based volume estimate (r=0.738 with target)
    'CalcVol_log',        # log-scaled formula value
    'CalcVol_vs_sph',     # formula residual vs spheroid estimate
    'Fill_Major',         # biochar/kiln size ratio  (MI=1.16)
    'Distance_norm',      # normalised distance       (MI=1.00)
    'Area_Ratio',         # 2-D fill fraction         (MI=0.97)
    'Log_Mean_Dist',      # log distance
    'Fill_x_Dist',        # fill × distance interaction
    'Biochar_AR',         # shape factor
    'Vol_spheroid',       # physics-based volume estimate (mm³)
    'Biochar_Minor_mm',   # absolute short-axis (mm)
    'Scaling_Factor',     # pixel→mm scale
    'Fill_Minor',         # minor-axis fill fraction
]

X_tree = Xeng[TREE_FEATURES]
print("\n[VIF]  Tree-model features (for transparency):")
print(compute_vif(X_tree).to_string(index=False))

# ──────────────────────────────────────────────────────────────
# 6. Train / Test Split  (stratified on volume bins)
# ──────────────────────────────────────────────────────────────
y_bins = pd.cut(y, bins=5, labels=False)
X_tr_lin,  X_te_lin,  y_tr, y_te = train_test_split(
    X_lin, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_bins)
idx_tr = X_tr_lin.index
idx_te = X_te_lin.index
X_tr_tree, X_te_tree = X_tree.loc[idx_tr], X_tree.loc[idx_te]

sc_lin  = RobustScaler()
X_tr_lin_s  = pd.DataFrame(sc_lin.fit_transform(X_tr_lin),   columns=X_lin.columns)
X_te_lin_s  = pd.DataFrame(sc_lin.transform(X_te_lin),       columns=X_lin.columns)

sc_tree = RobustScaler()
X_tr_tree_s = pd.DataFrame(sc_tree.fit_transform(X_tr_tree), columns=TREE_FEATURES)
X_te_tree_s = pd.DataFrame(sc_tree.transform(X_te_tree),     columns=TREE_FEATURES)

print(f"\n[SPLIT] Train: {len(y_tr)} | Test: {len(y_te)}  (stratified 80/20)")

# ──────────────────────────────────────────────────────────────
# 7. Model Suite
# ──────────────────────────────────────────────────────────────
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

linear_models = {
    'Ridge'     : Ridge(alpha=5.0),
    'Lasso'     : Lasso(alpha=0.5, max_iter=20000),
    'ElasticNet': ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=20000),
}

# Tree models — deliberately tuned for low overfitting (high min_samples_leaf)
tree_models = {
    'RandomForest'    : RandomForestRegressor(
        n_estimators=400, min_samples_leaf=5, max_features=0.8,
        n_jobs=-1, random_state=RANDOM_STATE),

    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=600, max_depth=4, learning_rate=0.035,
        subsample=0.75, min_samples_leaf=6,
        random_state=RANDOM_STATE),

    'HistGradBoosting': HistGradientBoostingRegressor(
        max_iter=700, max_depth=5, learning_rate=0.035,
        l2_regularization=0.3, min_samples_leaf=14,
        random_state=RANDOM_STATE),

    'ExtraTrees'      : ExtraTreesRegressor(
        n_estimators=400, min_samples_leaf=8, max_features=0.8,
        n_jobs=-1, random_state=RANDOM_STATE),

    'SVR'             : SVR(kernel='rbf', C=300, epsilon=8, gamma='scale'),

    'MLP'             : MLPRegressor(
        hidden_layer_sizes=(256, 128, 64), activation='relu',
        max_iter=2000, early_stopping=True, validation_fraction=0.1,
        learning_rate_init=0.001, random_state=RANDOM_STATE),
}

# Stacking ensemble
stacking_estimators = [
    ('gb',  GradientBoostingRegressor(n_estimators=600, max_depth=4,
                                       learning_rate=0.035, subsample=0.75,
                                       min_samples_leaf=6,
                                       random_state=RANDOM_STATE)),
    ('hgb', HistGradientBoostingRegressor(max_iter=700, max_depth=5,
                                            learning_rate=0.035,
                                            l2_regularization=0.3,
                                            min_samples_leaf=14,
                                            random_state=RANDOM_STATE)),
    ('et',  ExtraTreesRegressor(n_estimators=400, min_samples_leaf=8,
                                 max_features=0.8, n_jobs=-1,
                                 random_state=RANDOM_STATE)),
    ('svr', SVR(kernel='rbf', C=300, epsilon=8, gamma='scale')),
]
stacking_model = StackingRegressor(
    estimators=stacking_estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=5, n_jobs=-1,
)
tree_models['StackingEnsemble'] = stacking_model

# ──────────────────────────────────────────────────────────────
# 8. 10-Fold Cross-Validation
# ──────────────────────────────────────────────────────────────
def cv_eval(models_dict, X_s, y_arr, label=''):
    results = {}
    for name, model in models_dict.items():
        cv = cross_validate(model, X_s, y_arr, cv=kf,
                            scoring={'rmse': 'neg_root_mean_squared_error',
                                     'mae' : 'neg_mean_absolute_error',
                                     'r2'  : 'r2'},
                            return_train_score=True, n_jobs=-1)
        results[name] = {
            'CV_RMSE'    : -cv['test_rmse'].mean(),
            'CV_RMSE_std': cv['test_rmse'].std(),
            'CV_MAE'     : -cv['test_mae'].mean(),
            'CV_R2'      : cv['test_r2'].mean(),
            'CV_R2_std'  : cv['test_r2'].std(),
            'Tr_RMSE'    : -cv['train_rmse'].mean(),
            'Tr_R2'      : cv['train_r2'].mean(),
        }
        print(f"  {name:<20s}  RMSE={results[name]['CV_RMSE']:6.2f}±{results[name]['CV_RMSE_std']:.2f}"
              f"  MAE={results[name]['CV_MAE']:6.2f}"
              f"  R²={results[name]['CV_R2']:.4f}±{results[name]['CV_R2_std']:.4f}"
              f"  |  Tr_RMSE={results[name]['Tr_RMSE']:.2f}")
    return pd.DataFrame(results).T.sort_values('CV_RMSE')

print("\n" + "=" * 70)
print(f"10-FOLD CROSS-VALIDATION")
print("=" * 70)
print("── Linear models (VIF < 5 features) ──")
cv_lin = cv_eval(linear_models, X_tr_lin_s, y_tr.values)
print("\n── Tree / kernel / ensemble models (MI-curated features) ──")
cv_tree = cv_eval(tree_models, X_tr_tree_s, y_tr.values)
cv_all = pd.concat([cv_lin, cv_tree]).sort_values('CV_RMSE')

# ──────────────────────────────────────────────────────────────
# 9. Overfitting / Underfitting Diagnostics
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("OVERFITTING / UNDERFITTING DIAGNOSTICS")
print("=" * 70)
print(f"{'Model':<22s}  {'Tr RMSE':>9}  {'CV RMSE':>9}  {'Ratio':>7}  Status")
print("-" * 65)
for name, r in cv_all.iterrows():
    ratio  = r['Tr_RMSE'] / r['CV_RMSE']
    status = '⚠ OVERFIT'  if ratio < 0.65 else \
             '⚠ UNDERFIT' if r['CV_RMSE'] > 70 else '✓ OK'
    print(f"  {name:<20s}  {r['Tr_RMSE']:9.2f}  {r['CV_RMSE']:9.2f}  {ratio:7.3f}  {status}")

# ──────────────────────────────────────────────────────────────
# 10. Hyper-parameter Tuning — best model
# ──────────────────────────────────────────────────────────────
best_name  = cv_all.index[0]
print(f"\n[TUNE] Best by CV RMSE: {best_name}  ({cv_all.loc[best_name,'CV_RMSE']:.2f})")

# Tune only if it's a primary tree model (not stacking)
param_map = {
    'GradientBoosting': (
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        {'n_estimators':[500,700,900], 'max_depth':[3,4,5],
         'learning_rate':[0.025,0.035,0.05], 'subsample':[0.7,0.8],
         'min_samples_leaf':[4,6,10]}
    ),
    'HistGradBoosting': (
        HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        {'max_iter':[600,800,1000], 'max_depth':[4,5,6],
         'learning_rate':[0.025,0.035,0.05],
         'l2_regularization':[0.1,0.3,0.5],
         'min_samples_leaf':[10,14,20]}
    ),
    'RandomForest': (
        RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE),
        {'n_estimators':[300,500], 'max_depth':[None,25,35],
         'min_samples_leaf':[4,6,8], 'max_features':[0.7,0.8]}
    ),
    'SVR': (
        SVR(kernel='rbf'),
        {'C':[200,400,600], 'epsilon':[5,8,12], 'gamma':['scale','auto']}
    ),
    'ExtraTrees': (
        ExtraTreesRegressor(n_jobs=-1, random_state=RANDOM_STATE),
        {'n_estimators':[300,500], 'min_samples_leaf':[6,10,15],
         'max_features':[0.7,0.8,1.0]}
    ),
}

if best_name in param_map:
    base_est, params = param_map[best_name]
    rscv = RandomizedSearchCV(base_est, params, n_iter=30, cv=5,
                              scoring='neg_root_mean_squared_error',
                              n_jobs=-1, random_state=RANDOM_STATE)
    rscv.fit(X_tr_tree_s, y_tr.values)
    best_tuned = rscv.best_estimator_
    print(f"[TUNE] Best params : {rscv.best_params_}")
    print(f"[TUNE] 5-Fold RMSE : {-rscv.best_score_:.2f}")
else:
    # Stacking or unknown — use as-is
    best_tuned = tree_models[best_name]
    best_tuned.fit(X_tr_tree_s, y_tr.values)

# Validate tuned model with full 10-fold
cv_tuned = cross_validate(best_tuned, X_tr_tree_s, y_tr.values, cv=kf,
                           scoring={'rmse': 'neg_root_mean_squared_error',
                                    'mae' : 'neg_mean_absolute_error',
                                    'r2'  : 'r2'},
                           return_train_score=True, n_jobs=-1)
print(f"[TUNE] Tuned 10-Fold  RMSE: {-cv_tuned['test_rmse'].mean():.2f}±{cv_tuned['test_rmse'].std():.2f}"
      f"  R²: {cv_tuned['test_r2'].mean():.4f}")

# ──────────────────────────────────────────────────────────────
# 11. Leave-One-Out CV  (exact, Ridge linear baseline)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("LEAVE-ONE-OUT CV  (Ridge — VIF<5 features — training set only)")
print("=" * 70)
alphas   = np.logspace(-2, 5, 100)
ridge_gc = RidgeCV(alphas=alphas, gcv_mode='eigen')
ridge_gc.fit(X_tr_lin_s, y_tr.values)

loo      = LeaveOneOut()
loo_sc   = cross_val_score(Ridge(alpha=ridge_gc.alpha_), X_tr_lin_s, y_tr.values,
                            cv=loo, scoring='neg_root_mean_squared_error', n_jobs=-1)
loo_rmse = -loo_sc.mean()
print(f"  Ridge  α={ridge_gc.alpha_:.4f}")
print(f"  LOO RMSE = {loo_rmse:.2f} mL  (n={loo.get_n_splits(X_tr_lin_s)} splits)")

# ──────────────────────────────────────────────────────────────
# 12. Holdout Test Set Evaluation
# ──────────────────────────────────────────────────────────────
best_tuned.fit(X_tr_tree_s, y_tr.values)
y_pred_tr   = best_tuned.predict(X_tr_tree_s)
y_pred_te   = best_tuned.predict(X_te_tree_s)
residuals   = y_te.values - y_pred_te

tr_rmse  = np.sqrt(mean_squared_error(y_tr, y_pred_tr))
tr_r2    = r2_score(y_tr, y_pred_tr)
te_rmse  = np.sqrt(mean_squared_error(y_te, y_pred_te))
te_mae   = mean_absolute_error(y_te, y_pred_te)
te_r2    = r2_score(y_te, y_pred_te)
te_mape  = np.mean(np.abs(residuals / y_te.values)) * 100
cv_rmse  = -cv_tuned['test_rmse'].mean()
cv_r2    = cv_tuned['test_r2'].mean()

print("\n" + "=" * 70)
print(f"HOLDOUT TEST EVALUATION — {best_name} (tuned)")
print("=" * 70)
print(f"  Train      RMSE: {tr_rmse:.2f}   R²: {tr_r2:.4f}")
print(f"  Test       RMSE: {te_rmse:.2f}   MAE: {te_mae:.2f}   "
      f"MAPE: {te_mape:.2f}%   R²: {te_r2:.4f}")
print(f"  10-Fold CV RMSE: {cv_rmse:.2f}±{cv_tuned['test_rmse'].std():.2f}   "
      f"R²: {cv_r2:.4f}±{cv_tuned['test_r2'].std():.4f}")
print(f"  LOO (Ridge):     {loo_rmse:.2f} mL")

g_rmse = "✓ MET" if te_rmse <= 30 else ("~NEAR" if te_rmse <= 38 else "✗ NOT MET")
g_mae  = "✓ MET" if te_mae  <= 30 else ("~NEAR" if te_mae  <= 35 else "✗ NOT MET")
g_r2   = "✓ MET" if te_r2   >= 0.90 else "✗ NOT MET"
print(f"\n  Target RMSE ≤ 30 mL  : {g_rmse} ({te_rmse:.1f} mL)")
print(f"  Target MAE  ≤ 30 mL  : {g_mae}  ({te_mae:.1f} mL)")
print(f"  Target R²   ≥ 90 %   : {g_r2}  ({te_r2*100:.1f} %)")

# ──────────────────────────────────────────────────────────────
# 13. Feature Importance
# ──────────────────────────────────────────────────────────────
if hasattr(best_tuned, 'feature_importances_'):
    imp = pd.Series(best_tuned.feature_importances_, index=TREE_FEATURES).sort_values(ascending=False)
    print("\n[IMPORTANCE]  Feature importances:")
    for f, v in imp.items():
        bar = '█' * int(v * 50)
        print(f"  {f:<26s}  {v:.4f}  {bar}")

# ──────────────────────────────────────────────────────────────
# 14. Per-bin error
# ──────────────────────────────────────────────────────────────
print("\n[BIN]  Error by Volume bin (test set):")
bin_df = pd.DataFrame({'Actual': y_te.values, 'AbsErr': np.abs(residuals)})
print(bin_df.groupby('Actual')['AbsErr']
      .agg(MAE='mean', Std='std', Count='count').round(2).to_string())

# ──────────────────────────────────────────────────────────────
# 15. Diagnostic Plots
# ──────────────────────────────────────────────────────────────
print("\n[PLOT]  Generating diagnostic plots …")

# Scaled all-data version for learning curves
sc_all = RobustScaler()
X_all_s = pd.DataFrame(sc_all.fit_transform(X_tree), columns=TREE_FEATURES)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(f'Volume Prediction — {best_name} (tuned)  |  Test R²={te_r2:.4f}  RMSE={te_rmse:.1f}',
             fontsize=13, fontweight='bold')

# 15a. Learning curve
tr_sz, tr_sc, val_sc = learning_curve(
    best_tuned, X_all_s, y,
    cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='neg_root_mean_squared_error', n_jobs=-1)

ax = axes[0, 0]
ax.plot(tr_sz, -tr_sc.mean(1),  'o-', color='steelblue', lw=2, label='Train RMSE')
ax.plot(tr_sz, -val_sc.mean(1), 's-', color='coral',     lw=2, label='CV RMSE (5-fold)')
ax.fill_between(tr_sz,
                -tr_sc.mean(1)  - tr_sc.std(1),
                -tr_sc.mean(1)  + tr_sc.std(1),  alpha=0.12, color='steelblue')
ax.fill_between(tr_sz,
                -val_sc.mean(1) - val_sc.std(1),
                -val_sc.mean(1) + val_sc.std(1), alpha=0.12, color='coral')
ax.axhline(30, ls='--', color='green',  alpha=0.8, label='Target RMSE=30')
ax.axhline(25, ls=':',  color='darkgreen', alpha=0.6, label='Target RMSE=25')
ax.set_xlabel('Training samples'); ax.set_ylabel('RMSE (mL)')
ax.set_title('Learning Curve\n(gap = overfitting indicator)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 15b. Predicted vs Actual
ax = axes[0, 1]
ax.scatter(y_te, y_pred_te, alpha=0.35, s=14, color='steelblue', edgecolors='none')
lo, hi = 550, 1150
ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Perfect')
ax.fill_between([lo, hi], [lo-30, hi-30], [lo+30, hi+30],
                alpha=0.12, color='green', label='±30 mL band')
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel('Actual Volume (mL)'); ax.set_ylabel('Predicted (mL)')
ax.set_title(f'Predicted vs Actual — Test Set\nRMSE={te_rmse:.1f} | MAE={te_mae:.1f} | R²={te_r2:.4f}')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 15c. Residuals vs Predicted
ax = axes[0, 2]
ax.scatter(y_pred_te, residuals, alpha=0.35, s=12, color='steelblue', edgecolors='none')
ax.axhline(0,   color='red', lw=1.5, label='Zero error')
ax.axhline( 30, ls='--', color='orange', alpha=0.7)
ax.axhline(-30, ls='--', color='orange', alpha=0.7, label='±30 mL')
# Lowess trend
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    trend = lowess(residuals, y_pred_te, frac=0.3)
    ax.plot(trend[:,0], trend[:,1], 'r-', lw=2, label='LOWESS trend')
except Exception:
    pass
ax.set_xlabel('Predicted Volume (mL)'); ax.set_ylabel('Residual (mL)')
ax.set_title('Residuals vs Predicted\n(check for systematic bias)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 15d. Residual histogram + normality
ax = axes[1, 0]
ax.hist(residuals, bins=40, color='steelblue', edgecolor='white', alpha=0.8, density=True)
mu, sig = residuals.mean(), residuals.std()
xr = np.linspace(residuals.min(), residuals.max(), 200)
ax.plot(xr, stats.norm.pdf(xr, mu, sig), 'r-', lw=2, label=f'N(μ={mu:.1f}, σ={sig:.1f})')
ax.axvline(0, color='black', lw=1)
_, p_sw = stats.shapiro(np.random.choice(residuals, min(len(residuals), 5000), replace=False))
ax.set_xlabel('Residual (mL)'); ax.set_ylabel('Density')
ax.set_title(f'Residual Distribution\nMAE={te_mae:.1f} | Shapiro p={p_sw:.4f}')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 15e. VIF chart — linear features
ax = axes[1, 1]
vf = vif_lin.sort_values('VIF', ascending=True)
bar_c = ['#2ecc71' if v < 3 else '#f39c12' if v < 5 else '#e74c3c' for v in vf['VIF']]
ax.barh(vf['Feature'], vf['VIF'], color=bar_c, edgecolor='white')
ax.axvline(5, color='red',  ls='--', lw=1.5, label='VIF=5 threshold')
ax.axvline(3, color='gray', ls=':',  lw=1.0, label='VIF=3 (ideal)')
ax.set_xlabel('VIF'); ax.set_title('VIF — Linear Model Features\n(all < 5, collinearity resolved)')
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='x')

# 15f. Model comparison
ax = axes[1, 2]
all_names = list(cv_all.index)
all_rmse  = cv_all['CV_RMSE'].values
all_err   = cv_all['CV_RMSE_std'].values
bar_c2    = ['#2ecc71' if v <= 30 else '#f39c12' if v <= 40 else '#3498db'
             for v in all_rmse]
ax.barh(all_names[::-1], all_rmse[::-1], xerr=all_err[::-1],
        color=bar_c2[::-1], edgecolor='white', capsize=3)
ax.axvline(30, color='green', ls='--', lw=1.5, label='Target = 30')
ax.set_xlabel('10-Fold CV RMSE (mL)')
ax.set_title('Model Comparison — 10-Fold CV')
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plot_path = f'{OUTPUT_DIR}/volume_prediction_diagnostics.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {plot_path}")

# ──────────────────────────────────────────────────────────────
# 16. Final Summary
# ──────────────────────────────────────────────────────────────
vr = compute_vif(df[RAW_FEATURES]).set_index('Feature')
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"""
  ► DATA
    Samples: {len(df)}  |  Volume: 600–1100 mL (50 mL bins)  |  Missing: 0
    Discretisation noise floor: {df[TARGET].std()/np.sqrt(12):.1f} mL

  ► RAW FEATURE COLLINEARITY  (before engineering)
    Kiln_Major_Axis ↔ Scaling_Factor : r=0.984  VIF={vr.loc['Kiln_Major_Axis','VIF']:.0f} / {vr.loc['Scaling_Factor','VIF']:.0f}
    Kiln_Minor_Axis ↔ Biochar_Minor  : r=0.820  VIF={vr.loc['Kiln_Minor_Axis','VIF']:.0f}
    → Resolved via pixel→mm conversion + orthogonal feature design

  ► LINEAR-MODEL FEATURES  (VIF < 5, n={X_lin.shape[1]})""")
for f in X_lin.columns:
    print(f"    • {f:<28s}  VIF={vif_lin.set_index('Feature').loc[f,'VIF']:.2f}")
print(f"""
  ► TREE-MODEL FEATURES  (MI-ranked, n={len(TREE_FEATURES)})""")
for f in TREE_FEATURES:
    mi_v = mi_df.set_index('Feature').loc[f, 'MI']
    print(f"    • {f:<28s}  MI={mi_v:.3f}")
print(f"""
  ┌────────────────────────────────────────────────────────────┐
  │  BEST MODEL   : {best_name:<43s}│
  │  Test  RMSE   : {te_rmse:6.2f} mL                                    │
  │  Test  MAE    : {te_mae:6.2f} mL                                    │
  │  Test  R²     : {te_r2:6.4f}  ({te_r2*100:.1f} %)                          │
  │  Test  MAPE   : {te_mape:6.2f} %                                     │
  │  Train RMSE   : {tr_rmse:6.2f} mL  (overfit gap = {te_rmse-tr_rmse:.1f} mL)         │
  │  10-Fold RMSE : {cv_rmse:6.2f} ± {cv_tuned['test_rmse'].std():.2f} mL                         │
  │  10-Fold R²   : {cv_r2:6.4f} ± {cv_tuned['test_r2'].std():.4f}                          │
  │  LOO (Ridge)  : {loo_rmse:6.2f} mL                                    │
  ├────────────────────────────────────────────────────────────┤
  │  Target RMSE ≤ 30 mL  : {g_rmse:<37s}│
  │  Target MAE  ≤ 30 mL  : {g_mae:<37s}│
  │  Target R²   ≥ 90 %   : {g_r2:<37s}│
  │  Collinearity (VIF<5) : ✓ YES — all linear-model features     │
  └────────────────────────────────────────────────────────────┘""")
print("\nDone.")
